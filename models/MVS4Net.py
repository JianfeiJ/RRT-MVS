import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.module import *


class MVS4net(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d_hybrid', num_stage=4, fpn_base_channel=8,
                 reg_channel=8, stage_splits=[8, 8, 4, 4], depth_interals_ratio=[0.5, 0.5, 0.5, 0.5],
                 group_cor=False, group_cor_dim=[8, 8, 8, 8],
                 inverse_depth=False,
                 agg_type='ConvBnReLU3D',
                 attn_temp=2,
                 attn_fuse_d=False,
                 image_size=(832, 1152),
                 pe=True,
                 ):
        super(MVS4net, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.num_depth = [8, 8, 4, 4]
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.pe = pe
        if arch_mode == "fpn":
            self.feature = FPN(base_channels=fpn_base_channel, gn=False)

        # WSA setting
        self.img_size = (image_size[0] // 8, image_size[1] // 8)
        self.window_size = (image_size[0] // 64, image_size[1] // 64)
        self.depths = [[6], [6], [6], [6]]
        self.num_heads = [[4], [4], [4], [4]]

        self.stagenet = stagenet(inverse_depth, attn_fuse_d, attn_temp)

        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        if reg_net == 'reg3d' or 'reg3d_hybrid':
            self.down_size = [3, 3, 2, 2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]

            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d_hybrid':
                self.reg.append(reg2d_hybrid(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type,
                                             img_size=self.img_size, window_size=self.window_size,
                                             depths=self.depths[idx],
                                             num_heads=self.num_heads[idx], stage_idx=idx,
                                             pe_channel=self.stage_splits[idx],pe=self.pe))
            elif reg_net == 'reg3d_hybrid':
                self.reg.append(
                    reg3d_hybrid(in_channels=in_dim, base_channels=reg_channel, down_size=self.down_size[idx],
                                 img_size=self.img_size, window_size=self.window_size, depths=self.depths[idx],
                                 num_heads=self.num_heads[idx], stage_idx=idx))
            elif reg_net == 'reg2d':
                self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=in_dim, base_channels=reg_channel, down_size=self.down_size[idx]))

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()

        height_min, height_max = None, None
        width_min, width_max = None, None

        features = []
        for nview_idx in range(len(imgs)):  # imgs shape (B, N, C, H, W)
            img = imgs[nview_idx]
            features.append(self.feature(img))

        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B, C, H, W = features[0]['stage{}'.format(stage_idx + 1)].shape

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device,
                                                    imgs[0][0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], imgs[0][0].device,
                                            imgs[0][0].dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(),
                                                        outputs_stage['inverse_max_depth'].detach(),
                                                        self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_interval = (depth_max - depth_min) / 192
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx],
                                                self.depth_interals_ratio[stage_idx] * depth_interval, H, W)

            # Borrow from MVSFormer++ (https://github.com/maybeLx/MVSFormerPlusPlus)
            if self.pe:
                K = proj_matrices_stage[:, 0, 1, :3, :3]  # [B,3,3] stage1获取全局最小最大h,w,后续根据stage1的全局空间进行归一化
                depth_value = F.interpolate(depth_hypo, scale_factor=1 / (2 ** stage_idx), mode='bilinear',
                                            align_corners=True)
                _, _, h, w = depth_value.shape
                position3d, height_min, height_max, width_min, width_max = get_position_3d(
                    B, h, w, K, depth_value,
                    depth_min=depth_values.min(), depth_max=depth_values.max(),
                    height_min=height_min, height_max=height_max,
                    width_min=width_min, width_max=width_max,
                    normalize=True
                )
            else:
                position3d = None

            outputs_stage = self.stagenet(features_stage, proj_matrices_stage, depth_hypo=depth_hypo,
                                          regnet=self.reg[stage_idx], stage_idx=stage_idx,
                                          group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                          split_itv=self.depth_interals_ratio[stage_idx], position3d=position3d)

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


def cross_entropy_loss(mask_true, hypo_depth, depth_gt, attn_weight):
    B, D, H, W = attn_weight.shape
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6
    gt_index_image = torch.argmin(torch.abs(hypo_depth - depth_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W
    gt_index_volume = torch.zeros(B, D, H, W).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(attn_weight + 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    return masked_cross_entropy



def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)

        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss

    return total_loss, stage_ce_loss, range_err_ratio


def Blend_loss(inputs, depth_gt_ms, mask_ms, depth_min, depth_max, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1, 1, 1, 1])
    inverse = kwargs.get("inverse_depth", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ce_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        # mask range
        if inverse:
            depth_itv = (1 / hypo_depth[:, 2, :, :] - 1 / hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((1 / hypo_depth - 1 / depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        else:
            depth_itv = (hypo_depth[:, 2, :, :] - hypo_depth[:, 1, :, :]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(
                1) == 0  # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        # cross-entropy
        this_stage_ce_loss = cross_entropy_loss(mask, hypo_depth, depth_gt, attn_weight)

        stage_ce_loss.append(this_stage_ce_loss)
        total_loss = total_loss + this_stage_ce_loss

    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]
    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err3 = (abs_err <= 3).float().mean() * 100
    err1 = (abs_err <= 1).float().mean() * 100

    return total_loss, stage_ce_loss, range_err_ratio, epe, err3, err1
