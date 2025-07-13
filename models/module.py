import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import numpy as np
from .RRT import Transformer
from datasets.data_io import read_pfm, save_pfm
import cv2
import os
from abc import ABC


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs, Ws = src_fea.shape[-2:]
    B, num_depth, Hr, Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)], indexing='ij')
        y = y.reshape(Hr * Wr)
        x = x.reshape(Hr * Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth,
                                                                                               -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp == 0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    if len(src_fea.shape) == 4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape) == 5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(
                F.grid_sample(src_fea[:, :, d], grid.reshape(B, num_depth, Hr, Wr, 2)[:, d], mode='bilinear',
                              padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea


def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                     requires_grad=False).reshape(1,
                                                                                                  -1) * new_interval.squeeze(
        1))  # (B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, D, H, W)
    return depth_range_samples


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H,
                                                                                                                W) / (
                  ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None,
                                                                  None] * itv

    return 1. / inverse_depth_hypo


def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :,
                                                            :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear',
                                       align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo


def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:, None, None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:, None, None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (
            torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                         requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear',
                                        align_corners=True).squeeze(1)
    return depth_range_samples


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def convbnrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           output_padding=output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return



class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu



class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D'):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1, 3, 3),
                                                       pad=(0, 1, 1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel * 2, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv2 = getattr(module, conv_name)(base_channel * 2, base_channel * 2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel * 2, base_channel * 4, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv4 = getattr(module, conv_name)(base_channel * 4, base_channel * 4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel * 4, base_channel * 8, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv6 = getattr(module, conv_name)(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x, position3d=None):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)


class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels * 4, base_channels * 4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels * 8, base_channels * 8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, position3d=None):
        if self.down_size == 3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size == 2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x.squeeze(1)  # B D H W

class reg3d_hybrid(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3, img_size=(64, 80),
                 window_size=(16, 20), depths=[2], num_heads=[4], stage_idx=0):
        super(reg3d_hybrid, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels * 4, base_channels * 4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels * 8, base_channels * 8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, output_padding=1,
                                   stride=2, bias=False),
                nn.BatchNorm3d(base_channels * 2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))

        self.stage_idx = stage_idx
        self.img_size = img_size
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads

        self.down = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 8, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
                      stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
            LayerNorm3D(base_channels * 8, eps=1e-6)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
                               stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
            LayerNorm3D(base_channels, eps=1e-6)
        )

        self.RRT = Transformer(img_size=self.img_size, in_chans=base_channels,
                               depths=self.depths, num_heads=self.num_heads,
                               window_size=self.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, patch_norm=True, input_channel=in_channels)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, position3d=None):
        if self.down_size == 3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)

            x = self.down(x)
            x = self.RRT(x, position3d)
            x = self.up(x)

            x = self.prob(x)
        elif self.down_size == 2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)

            x = self.down(x)
            x = self.RRT(x, position3d)
            x = self.up(x)

            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)

            x = self.down(x)
            x = self.RRT(x, position3d)
            x = self.up(x)

            x = self.prob(x)
        return x.squeeze(1)  # B D H W



class reg2d_hybrid(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', img_size=(64, 80),
                 window_size=(16, 20), depths=[2], num_heads=[4], stage_idx=0, pe_channel=8, pe=True):
        super(reg2d_hybrid, self).__init__()
        module = importlib.import_module("models.module")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1, 3, 3),
                                                       pad=(0, 1, 1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel * 2, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv2 = getattr(module, conv_name)(base_channel * 2, base_channel * 2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel * 2, base_channel * 4, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv4 = getattr(module, conv_name)(base_channel * 4, base_channel * 4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel * 4, base_channel * 8, kernel_size=(1, 3, 3),
                                                       stride=(1, 2, 2), pad=(0, 1, 1))
        self.conv6 = getattr(module, conv_name)(base_channel * 8, base_channel * 8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                               output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.stage_idx = stage_idx
        self.img_size = img_size
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads

        self.down = nn.Sequential(
            nn.Conv3d(base_channel, base_channel * 8, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
                      stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
            LayerNorm3D(base_channel * 8, eps=1e-6)
        )

        self.up = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
                               stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
            LayerNorm3D(base_channel, eps=1e-6)
        )

        self.RRT = Transformer(img_size=self.img_size, in_chans=base_channel,
                                            depths=self.depths, num_heads=self.num_heads,
                                            window_size=self.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                            norm_layer=nn.LayerNorm, patch_norm=True, input_channel=pe_channel, pe=pe)

        # self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

        self.prob = nn.Sequential(
            nn.Conv3d(16, 8, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1, stride=1, padding=0)
        )
        # self.depth_residual_attn = DRA(base_channel, base_channel)
        # self.fusion = ConvBnReLU3D(16, 8)

    def forward(self, x, position3d=None):

        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x1 = self.conv6(self.conv5(conv4))
        x1 = conv4 + self.conv7(x1)
        x1 = conv2 + self.conv9(x1)
        x1 = conv0 + self.conv11(x1)

        # x = self.depth_residual_attn(x)

        # if position3d is not None:
        #     if self.use_pe_proj:
        #         x = x + self.pe_proj(PositionEncoding3D(position3d, x.shape[1]))  # position encoding
        #     else:
        #         x = x + self.pe_proj(PositionEncoding3D(position3d, x.shape[1] // 3))  # position encoding

        x2 = self.down(conv0)
        x2 = self.RRT(x2, position3d)
        x2 = self.up(x2)

        # x = self.depth_residual_attn(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.prob(x)

        del x1, x2

        return x.squeeze(1)


# class reg2d_hybrid(nn.Module):
#     def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D', img_size=(64, 80),
#                  window_size=(16, 20), depths=[2], num_heads=[4], stage_idx=0):
#         super(reg2d_hybrid, self).__init__()
#         module = importlib.import_module("models.module")
#         stride_conv_name = 'ConvBnReLU3D'
#         self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1, 3, 3),
#                                                        pad=(0, 1, 1))
#         self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel * 2, kernel_size=(1, 5, 5),
#                                                        stride=(1, 2, 2), pad=(0, 2, 2))
#         self.conv2 = getattr(module, conv_name)(base_channel * 2, base_channel * 2, kernel_size=(5, 3, 3), pad=(2, 1, 1))
#
#         self.conv3 = getattr(module, stride_conv_name)(base_channel * 2, base_channel * 4, kernel_size=(1, 5, 5),
#                                                        stride=(1, 2, 2), pad=(0, 2, 2))
#         self.conv4 = getattr(module, conv_name)(base_channel * 4, base_channel * 4, kernel_size=(5, 3, 3), pad=(2, 1, 1))
#
#         self.conv5 = getattr(module, stride_conv_name)(base_channel * 4, base_channel * 8, kernel_size=(1, 5, 5),
#                                                        stride=(1, 2, 2), pad=(0, 2, 2))
#         self.conv6 = getattr(module, conv_name)(base_channel * 8, base_channel * 8, kernel_size=(5, 3, 3), pad=(2, 1, 1))
#
#         self.conv7 = nn.Sequential(
#             nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 5, 5), padding=(0, 2, 2),
#                                output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
#             nn.BatchNorm3d(base_channel * 4),
#             nn.ReLU(inplace=True))
#
#         self.conv9 = nn.Sequential(
#             nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 5, 5), padding=(0, 2, 2),
#                                output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
#             nn.BatchNorm3d(base_channel * 2),
#             nn.ReLU(inplace=True))
#
#         self.conv11 = nn.Sequential(
#             nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 5, 5), padding=(0, 2, 2),
#                                output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
#             nn.BatchNorm3d(base_channel),
#             nn.ReLU(inplace=True))
#
#         self.stage_idx = stage_idx
#         self.img_size = img_size
#         self.window_size = window_size
#         self.depths = depths
#         self.num_heads = num_heads
#
#         self.down = nn.Sequential(
#             nn.Conv3d(base_channel, base_channel * 8, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
#                       stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
#             LayerNorm3D(base_channel * 8, eps=1e-6)
#         )
#
#         self.up = nn.Sequential(
#             nn.ConvTranspose3d(base_channel * 8, base_channel, kernel_size=(1, 2 ** stage_idx, 2 ** stage_idx),
#                                stride=(1, 2 ** stage_idx, 2 ** stage_idx)),
#             LayerNorm3D(base_channel, eps=1e-6)
#         )
#
#         self.RRT = Transformer(img_size=self.img_size, in_chans=base_channel,
#                                             depths=self.depths, num_heads=self.num_heads,
#                                             window_size=self.window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                                             norm_layer=nn.LayerNorm, patch_norm=True, input_channel=input_channel)
#
#         self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)
#
#     def forward(self, x, position3d=None):
#
#         conv0 = self.conv0(x)
#         conv2 = self.conv2(self.conv1(conv0))
#         conv4 = self.conv4(self.conv3(conv2))
#         x = self.conv6(self.conv5(conv4))
#         x = conv4 + self.conv7(x)
#         x = conv2 + self.conv9(x)
#         x = conv0 + self.conv11(x)
#
#         x = self.down(x)
#         x = self.RRT(x, position3d)
#         x = self.up(x)
#
#         x = self.prob(x)
#
#         return x.squeeze(1)


class FPN(nn.Module):
    """
    FPN aligncorners downsample 4x"""

    def __init__(self, base_channels, gn=False, dcn=False):
        super(FPN, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.dcn = dcn
        # if self.dcn:
        #     self.dcn1 = NA_DCN(base_channels * 8, 3, gn=gn)
        #     self.dcn2 = NA_DCN(base_channels * 4, 3, gn=gn)
        #     self.dcn3 = NA_DCN(base_channels * 2, 3, gn=gn)
        #     self.dcn4 = NA_DCN(base_channels * 1, 3, gn=gn)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv2)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv1)
        out3 = self.out3(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv0)
        out4 = self.out4(intra_feat)

        if self.dcn:
            out1 = self.dcn1(out1)
            out2 = self.dcn2(out2)
            out3 = self.dcn3(out3)
            out4 = self.dcn4(out4)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3
        outputs["stage4"] = out4

        return outputs


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def depth_wta(p, depth_values):
    '''Winner take all.'''
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map


def get_position_3d(B, H, W, K, depth_values, depth_min, depth_max, height_min, height_max, width_min, width_max, normalize=True):
    num_depth = depth_values.shape[1]
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=K.device),
                               torch.arange(0, W, dtype=torch.float32, device=K.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(torch.inverse(K), xyz)  # [B, 3, H*W]
        # point position in 3d space
        position3d = xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(B, 1, num_depth, -1)  # [B, 3, 1, H*W]->[B,3,D,H*W]
        if normalize:  # minmax normalization
            if height_min is None or height_max is None or width_min is None or width_max is None:
                width_min, width_max = position3d[:, 0].min(), position3d[:, 0].max()
                height_min, height_max = position3d[:, 1].min(), position3d[:, 1].max()
            position3d[:, 0] = (position3d[:, 0] - width_min) / (width_max - width_min + 1e-5)
            position3d[:, 1] = (position3d[:, 1] - height_min) / (height_max - height_min + 1e-5)
            # normalizing depth (z) into 0~1 based on depth_min and depth_max
            position3d[:, 2] = (torch.clamp(position3d[:, 2], depth_min, depth_max) - depth_min) / (depth_max - depth_min + 1e-5)

        position3d = position3d.reshape(B, 3, num_depth, H, W)

    return position3d, height_min, height_max, width_min, width_max

class stagenet(nn.Module):
    def __init__(self, inverse_depth=False, attn_fuse_d=True, attn_temp=2):
        super(stagenet, self).__init__()
        self.inverse_depth = inverse_depth
        self.attn_fuse_d = attn_fuse_d
        self.attn_temp = attn_temp


    def forward(self, features, proj_matrices, depth_hypo, regnet, stage_idx, group_cor=False, group_cor_dim=8,
                split_itv=1, position3d=None):

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B, D, H, W = depth_hypo.shape
        C = ref_feature.shape[1]

        cor_weight_sum = 1e-8
        cor_feats = 0
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo)  # B C D H W

            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C // group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src) ** 2  # B C D H W
            del warped_src, src_proj, src_fea

            if not self.attn_fuse_d:
                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W

                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
            else:
                cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                cor_weight_sum += cor_weight  # B D H W
                cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B G D H W
            del cor_weight, cor_feat
        if not self.attn_fuse_d:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B G D H W

        del cor_weight_sum, src_features

        # step 3. regularization
        attn_weight = regnet(cor_feats, position3d)  # B D H W
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W

        with torch.no_grad():
            photometric_confidence = attn_weight.max(1)[0]  # B H W
            photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),
                                                   scale_factor=2 ** (3 - stage_idx), mode='bilinear',
                                                   align_corners=True).squeeze(1)

        ret_dict = {"depth": depth, "photometric_confidence": photometric_confidence,
                    "hypo_depth": depth_hypo, "attn_weight": attn_weight}

        if self.inverse_depth:
            last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
            inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth

        return ret_dict
