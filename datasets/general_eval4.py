from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *
from copy import deepcopy

s_h, s_w = 0, 0


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = 192  # Hardcode
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  # whether to fix the resolution of input image.
        self.fix_wh = False

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale_h = 1.0 * max_h / h
            scale_w = 1.0 * max_w / w
            # if scale * w > max_w:
            #     scale = 1.0 * max_w / w
            new_w, new_h = scale_w * w // base * base, scale_h * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def mark(self, flag, k, b, t, x, y, thresh=1, inv=0):
        H, W = flag.shape
        if inv:
            x_pred = k * y + b
            delta = (abs(x - x_pred) < thresh).reshape(H, W)
        else:
            y_pred = k * x + b
            delta = (abs(y - y_pred) < thresh).reshape(H, W)
        idx = np.where(delta)
        idx_y, idx_x = idx
        if not len(idx_y):
            return flag
        flag[idx_y, idx_x] = t
        return flag

    def epipolar_search(self, rot, trans, H, W):
        t1, t2, t3 = trans[0], trans[1], trans[2]
        f = np.zeros((H, W, 2))
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x, y = x.reshape(-1), y.reshape(-1)
        grid = np.stack((x, y, np.ones_like(x)), axis=0)
        A = np.matmul(rot, grid).reshape(3, H, W)
        inv = 0
        f[:, :, 0] = (A[1, :] * t3 - A[2, :] * t2) / (A[0, :] * t3 - A[2, :] * t1 + 1e-10)
        f[:, :, 1] = (t2 / (t3 + 1e-10)) - f[:, :, 0] * (t1 / (t3 + 1e-10))
        if abs(f[:, :, 0]).max() >= 5:
            f[:, :, 0] = 1. / (f[:, :, 0] + 1e-10)
            f[:, :, 1] = (t1 / (t3 + 1e-10)) - f[:, :, 0] * (t2 / (t3 + 1e-10))
            inv = 1
        f[:, :, 0] = 0.1 * np.floor(10 * f[:, :, 0])
        f[:, :, 1] = 10 * np.floor(0.1 * f[:, :, 1])
        k_all = np.unique(f[:, :, 0])
        b_all = np.unique(f[:, :, 1])
        ref_flag = np.zeros((H, W))
        src_flag = np.zeros((H, W))
        t = 1

        if len(k_all) > 500 or len(b_all) > 500:
            return ref_flag, src_flag

        for k in k_all:
            for b in b_all:
                idx = np.where(np.logical_and(f[:, :, 0] == k, f[:, :, 1] == b))
                if len(idx[0]):
                    src_flag = self.mark(src_flag, k, b, t, x, y, inv=inv)
                    idx_y, idx_x = idx
                    ref_flag[idx_y, idx_x] = t
                    t += 1

        return ref_flag, src_flag

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta

        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
            self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            imgs.append(img.transpose(2, 0, 1))
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

        # all
        # imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        poses = deepcopy(stage1_pjmats)
        ref_proj, src_projs = poses[0], poses[1:]
        ref_proj_new = ref_proj[0].copy()
        ref_proj_new[:3, :4] = np.matmul(ref_proj[1, :3, :3], ref_proj[0, :3, :4])
        H, W = imgs[0].shape[-2:]
        for i, src_proj in enumerate(src_projs):
            src_proj_new = src_proj[0].copy()
            src_proj_new[:3, :4] = np.matmul(src_proj[1, :3, :3], src_proj[0, :3, :4])
            proj = np.matmul(src_proj_new, np.linalg.inv(ref_proj_new))
            rot = proj[:3, :3]
            trans = proj[:3, 3:4].reshape(-1)
            ref_flag, src_flag = self.epipolar_search(rot, trans, H // 8, W // 8)
            flag_rs = np.stack((ref_flag, src_flag), axis=0)[None, :, :, :]
            if i == 0:
                flag = flag_rs
            else:
                flag = np.concatenate((flag, flag_rs), axis=0)

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": proj_matrices,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats
        }
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                'flag': flag}