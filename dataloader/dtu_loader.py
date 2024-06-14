import os, time
import imageio
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.append("./")
sys.path.append("../")
from dataloader.utils.ray_voxel_intersection import ray_voxel_intersect
from dataloader.sampler import GridPointsSampler
from run_nerf_helpers import get_rays_np
import utils.camera as camera
from dataloader.common import VoxelStorageDS, single_proc_write

class DTUDatasetBase(Dataset): 
    def __init__(self, basedir, name, 
        n_train_imgs, 
        split_type="pixelnerf",
        no_test_rest=True, 
        mask_path=None, 
        voxel_num=64, 
        radius_div=4, 
        scene_pad=0, 
        chunk=32*1024, 
        n_sur_pts=9, 
        n_nxt_ray_pts=9, 
        pre_save_dir="vox_ray_storage/", 
        eval=False): 

        pass

    def load_dtu_data(self, 
        path, mask_path=None, 
        split_type="pixelnerf", 
        n_input_views=-1, 
        no_test_rest=True): 
        raise raiseNotImplementedError("load_dtu_data is not implemented. ")
    
    def print_range(self, near, far, rays_o, rays_d): 
        """a reference of how to decide scene_range, though it is hard-coded"""
        rays_near = rays_o + rays_d * near * np.ones_like(rays_d)
        rays_far = rays_o + rays_d * far * np.ones_like(rays_d)
        # (N*HW, 3)
        near_xmin = np.min(rays_near[..., 0])
        near_xmax = np.max(rays_near[..., 0])
        near_ymin = np.min(rays_near[..., 1])
        near_ymax = np.max(rays_near[..., 1])
        near_zmin = np.min(rays_near[..., 2])
        near_zmax = np.max(rays_near[..., 2])
        far_xmin = np.min(rays_far[..., 0])
        far_xmax = np.max(rays_far[..., 0])
        far_ymin = np.min(rays_far[..., 1])
        far_ymax = np.max(rays_far[..., 1])
        far_zmin = np.min(rays_far[..., 2])
        far_zmax = np.max(rays_far[..., 2])

        dist = rays_far - rays_near
        x_range = np.abs(dist[:, 0]).max()
        y_range = np.abs(dist[:, 1]).max()
        z_range = np.abs(dist[:, 2]).max()

        print("near: (xmin, xmax): ({}, {}), (ymin, ymax): ({}, {}), (zmin, zmax): ({}, {})".format(
            near_xmin, near_xmax, near_ymin, near_ymax, near_zmin, near_zmax))
        print("far: (xmin, xmax): ({}, {}), (ymin, ymax): ({}, {}), (zmin, zmax): ({}, {})".format(
            far_xmin, far_xmax, far_ymin, far_ymax, far_zmin, far_zmax))
        print("xrange: {}, yrange: {}, zrange: {}".format(x_range, y_range, z_range))
        
        return
        
    def get_scene_range(self, near, far): 
        """a reference of how to decide scene_range, though it is hard-coded"""
        raise raiseNotImplementedError("get_scene_range is not implemented. ")
    
    def get_ray_voxel_intersect(self, chunk):
        """only supports gpu processing, only supports all images with same resolution"""

        print("-"*10+"Getting the intersection between rays and voxels..."+"-"*10)
        rays_o = torch.Tensor(self.rays_o).to(self.device)
        rays_d = torch.Tensor(self.rays_d).to(self.device)
        print("number of rays: {}".format(rays_o.shape))

        voxel_flag = [VoxelStorageDS() for _ in range(self.voxel_num**3)]
        start_all = time.time()
        # assume all rays will hit certain number of voxels
        for i in range(0, rays_o.shape[0], chunk):
            # depends on GPU memory available
            n_rays = min(i+chunk, rays_o.shape[0]) - i
            coord, mask, ts = ray_voxel_intersect(
                rays_o[i:i+n_rays], rays_d[i:i+n_rays], self.xyzmin, self.xyzmax, self.voxel_num, self.voxel_size
            )

            ts_in = ts[:, :-1, None]
            ts_out = ts[:, 1:, None]
            ts = ts[:,:-1]
            coord = coord.clamp_min(0)
            
            mask = mask[:,:-1]&mask[:,1:]
            # NOTE: the return coord refers to the coord w.r.t voxel, 
            # need to transform it back
            coord_in_int = coord[:,:-1][mask]
            coord_out_int = coord[:,1:][mask]
            coord = coord * self.voxel_size + self.xyzmin
            coord_in = coord[:,:-1][mask]
            coord_out = coord[:,1:][mask]
            ts_in = ts_in[mask]
            ts_out = ts_out[mask]
            
            n_max_intersect = coord.shape[1] - 1
            rays_idx = torch.arange(i, i+n_rays).view(n_rays, -1, 1).expand(n_rays, n_max_intersect, 1) # (N, max_intersect_voxels, 1)
            rays_idx = rays_idx.to(self.device)
            rays_idx = rays_idx[mask]
            ts_flat = ts[..., None][mask]
            # ts (t of the intersected planes)/mask, (N, max_intersect_voxels)
            # coord, (N, max_intersect_voxels, 3)
            # coord_in/coord_out, (all_intersected_voxels, 3)
            # rays_idx, (all_intersected_voxels, 1)
            # ts_flat, (all_intersected_voxels, 1)

            target_rgb = torch.Tensor(self.rays_rgb).to(self.device)
            target_rgb = target_rgb[rays_idx.view(-1).long()]
            voxel_info = torch.cat([rays_idx, coord_in, coord_out, ts_in, ts_out, target_rgb], -1) # (all_intersected_voxels, 5)
            
            pmin = torch.min((coord_in_int+1e-4).long(),(coord_out_int+1e-4).long())
            xmin, ymin, zmin = pmin.clamp_max(self.voxel_num-1).split(1,dim=-1) # index of the intersected voxels, (all_intersected_voxels, 1)
            start = time.time()
            voxel_info = voxel_info.cpu().numpy()
            xmin, ymin, zmin = [m.view(-1).cpu().numpy().tolist() for m in [xmin, ymin, zmin]]
            for j in range(len(xmin)):
                voxel_flag[xmin[j]*(self.voxel_num**2)+ymin[j]*(self.voxel_num)+zmin[j]].add(*voxel_info[j])
    
            print("Processing {} voxels with {} time, idx: {}.".format(pmin.shape[0], time.time()-start, i))
        
        print("get_ray_voxel_intersect function of {} images costs {} time. ".format(self.imgs.shape[0], time.time()-start_all))
        print("-"*10+"Done"+"-"*10)
        return voxel_flag, np.array([vf.get_n_rays() for vf in voxel_flag])
    
    def __len__(self):
        return np.sum(self.in_voxel_sum)
    
    def get_record(self, voxel_idx, cursor_in_voxel, nxt_ray_idx, vox_end):
        file_name = os.path.join(self.save_dir, self.prefix+"_ray_{}.npy".format(voxel_idx))
        record = np.load(file_name)
        return record[cursor_in_voxel], record[nxt_ray_idx]
    
    def data_from_record(self, voxel_idx, record, n_sample_pts):
        xin, yin, zin, xout, yout, zout = record[1:7]
        ts_in, ts_out = record[7:9]
        r, g, b = record[9:12]

        ref_pt = np.array([(xin+xout)/2., (yin+yout)/2., (zin+zout)/2.]).astype(np.float32)
        t_ref_pt = np.array([(ts_in+ts_out)/2.]).astype(np.float32)
        ray_idx = int(record[0])
        ray_o = self.rays_o[ray_idx]
        ray_d = self.rays_d[ray_idx]
        ray_rgb = self.rays_rgb[ray_idx]

        # sampling around the ref_pt
        sur_pts = np.stack([ref_pt, ref_pt], axis=0)
        sur_pts = self.grid_sampler.sphere_sample(sur_pts, self.voxel_size/self.radius_div, self.n_sur_pts)
        sur_pts = sur_pts[0].transpose(1, 0).astype(np.float32)

        return ray_o, ray_d, ray_rgb, ref_pt, t_ref_pt, sur_pts
    
    def data_from_nxt_ray_record(self, record, n_sample_pts):
        xin, yin, zin, xout, yout, zout = record[1:7]
        ts_in, ts_out = record[7:9]
        
        rand = np.random.rand(n_sample_pts)
        x = xin + (xout-xin)*rand
        y = yin + (yout-yin)*rand
        z = zin + (zout-zin)*rand
        t = (ts_in + (ts_out-ts_in)*rand).astype(np.float32)
        nxt_ray_pts = np.stack([x, y, z], axis=0).astype(np.float32)

        return nxt_ray_pts, t
    
    def __getitem__(self, idx):
        """return a ray, and sampled points of the sampled voxel"""
        voxel_idx, cursor_in_voxel, nxt_ray_idx, vox_end = idx[0], idx[1], idx[2], idx[3]
        if self.voxel_flag is None:
            record, record_nxt_ray = self.get_record(voxel_idx, cursor_in_voxel, nxt_ray_idx, vox_end)
        else:
            pass
            # record = self.voxel_flag[voxel_idx].get_info()[cursor_in_voxel]

        ray_o, ray_d, ray_rgb, ref_pt, t_ref_pt, sur_pts = self.data_from_record(voxel_idx, record, self.n_sur_pts)

        nxt_ray_pts, nxt_ray_t_pts = self.data_from_nxt_ray_record(record, self.n_nxt_ray_pts)
        # (3, n_nxt_ray_pts), (n_nxt_ray_pts, )
        vox_end = np.array([vox_end]).astype(np.float32)

        return ray_o, ray_d, ray_rgb, ref_pt, t_ref_pt, \
                np.concatenate([sur_pts, nxt_ray_pts], axis=1), \
                nxt_ray_t_pts, vox_end
        
class DTUDatasetDefault(DTUDatasetBase): 
    def __init__(self, basedir, name, 
        n_train_imgs, 
        split_type="pixelnerf",
        no_test_rest=True, 
        mask_path=None, 
        voxel_num=64, 
        radius_div=4, 
        scene_pad=0, 
        chunk=32*1024, 
        n_sur_pts=9, 
        n_nxt_ray_pts=9, 
        pre_save_dir="vox_ray_storage/", 
        eval=False): 

        self.dataset = "DTU_{}".format(name)
        self.device = torch.device("cuda")
        
        assert n_train_imgs in [3, 6, 9]
        self.n_train_imgs = n_train_imgs
        self.n_sur_pts = n_sur_pts
        self.n_nxt_ray_pts = n_nxt_ray_pts
        self.voxel_num = voxel_num
        self.radius_div = radius_div
        
        self.imgs, self.poses, self.hwf, self.test_masks, self.i_split = \
            self.load_dtu_data(basedir, mask_path, split_type, n_train_imgs, no_test_rest)
        
        self.render_poses = self.generate_render_poses(c2ws=self.poses, N_views=6)
        self.i_train, self.i_val, self.i_test = self.i_split
        self.near, self.far = 0.5, 3.5 # Default config

        print('Loaded DTU Default ', self.imgs.shape, self.poses.shape, self.render_poses.shape, self.hwf, basedir)
        
        H, W, focal = self.hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]])
        
        self.rays_o, self.rays_d, self.scene_range = self.get_scene_range(self.near, self.far)
        self.scene_range += scene_pad

        select_train_imgs = np.stack(
            [self.imgs[self.i_train[i]] for i in range(self.n_train_imgs)], 0)
        self.rays_rgb = select_train_imgs.reshape(-1, 3)

        self.voxel_size = self.scene_range / self.voxel_num
        self.xyzmin, self.xyzmax = -0.5*self.scene_range, 0.5*self.scene_range
    
        self.save_dir = pre_save_dir+"{}_voxel_flag_{}x{}_{}vox_range{}_{}images_defaultreader".format(
            self.dataset, H, W, self.voxel_num, str(self.scene_range), self.n_train_imgs)
        self.prefix = "{}x{}_{}vox_range{}_{}images_defaultreader".format(
            H, W, self.voxel_num, str(self.scene_range), self.n_train_imgs)
        print("Save_dir: {}, prefix: {}".format(self.save_dir, self.prefix))
        self.voxel_flag, self.in_voxel_num = None, None
        if not eval: 
            if not os.path.exists(self.save_dir):
                self.voxel_flag, self.in_voxel_sum = self.get_ray_voxel_intersect(chunk)
                single_proc_write(self.save_dir, self.prefix, self.voxel_flag, self.in_voxel_sum)
            else: 
                self.in_voxel_sum = np.load(os.path.join(self.save_dir, self.prefix+"_voxel_sum.npy"))

        self.grid_sampler = GridPointsSampler(self.n_sur_pts, self.voxel_size, "split_region")

    def get_scene_range(self, near, far): 
        """a reference of how to decide scene_range, though it is hard-coded"""
        rays = np.stack([get_rays_np(self.hwf[0], self.hwf[1], self.K, p) \
                            for p in self.poses[:,:3,:4]], 0)
        # [N, ro+rd, H, W, 3]
        # use all images to get the bound, 
        # but only return n_train imgs for training.
        rays_o, rays_d = (rays[:, 0]).reshape(-1, 3), (rays[:, 1].reshape(-1, 3))
        
        self.print_range(near, far, rays_o, rays_d)

        scene_range = 6

        rays_o = np.concatenate(
            [rays_o[(self.i_train[i]*self.hwf[0]*self.hwf[1]):((self.i_train[i]+1)*self.hwf[0]*self.hwf[1])] for i in range(self.n_train_imgs)], 0)
        rays_d = np.concatenate(
            [rays_d[(self.i_train[i]*self.hwf[0]*self.hwf[1]):((self.i_train[i]+1)*self.hwf[0]*self.hwf[1])] for i in range(self.n_train_imgs)], 0)

        return rays_o, rays_d, scene_range
    
    def get_loaded_info(self):
        return self.imgs, self.test_masks, self.poses, \
                self.render_poses, self.hwf, self.i_split
    
    def load_dtu_data(self, 
        path, mask_path=None, 
        split_type="pixelnerf", 
        n_input_views=-1, 
        no_test_rest=True): 
        """Default dtu loader"""

        imgdir = os.path.join(path, 'image')
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        imgnames = [f'{int(i.split("/")[-1].split(".")[0]):03d}.png' for i in imgfiles]

        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

        imgs = []
        for fname in imgfiles: 
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
            imgs.append(image)

        imgs = np.stack(imgs, 0)
        num = imgs.shape[0]
        
        cam_path = os.path.join(path, "cameras.npz")
        all_cam = np.load(cam_path)

        focal = 0

        coord_trans_world = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )
        coord_trans_cam = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )

        poses = []
        for i in range(num):
            P = all_cam["world_mat_" + str(i)]
            P = P[:3]

            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]

            focal += (K[0,0] + K[1,1]) / 2

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]

            scale_mtx = all_cam.get("scale_mat_" + str(i))
            if scale_mtx is not None:
                norm_trans = scale_mtx[:3, 3:]
                norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                pose[:3, 3:] -= norm_trans
                pose[:3, 3:] /= norm_scale

            pose = (
                    coord_trans_world
                    @ pose
                    @ coord_trans_cam
                )
            poses.append(pose[:3,:4])
        
        poses = np.stack(poses)
        print('poses shape:', poses.shape)


        focal = focal / num
        H, W = imgs[0].shape[:2]
        print("HWF", H, W, focal)

        print("split_type: {}.".format(split_type))
        if split_type == "pixelnerf":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
        elif split_type == "pixelnerf_reduced_testset":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13, 24, 30, 41, 47, 43, 29, 45,
                    34, 33]
            test_idx = [1, 2, 9, 10, 11, 12, 14, 15, 23, 26, 27, 31, 32, 35, 42, 46]
    
        if n_input_views != -1:
            train_idx = train_idx[:n_input_views]
            if not no_test_rest:
                # InfoNeRF config
                test_idx = [i for i in range(49) if i not in train_idx]

        val_idx = test_idx

        # mask_path
        # ../data/DTU_mask/scan114
        masks = None
        if mask_path is not None:
            if "mask" in os.listdir(mask_path):
                mask_path = os.path.join(mask_path, "mask")
            test_imgnames = [imgnames[i] for i in test_idx]
            maskfiles = [os.path.join(mask_path, name) for name in test_imgnames] 
        if mask_path is not None:
            # masks = [imread(f)[...,:3]/255. for f in maskfiles]
            masks = []
            for f in maskfiles:
                with open(f, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32)[:, :, :3]/255.
                    image = (image == 1).astype(np.float32)
                masks.append(image)
            masks = np.stack(masks, 0)

        resz_masks = np.zeros((masks.shape[0], H, W, 3))
        for i in range(masks.shape[0]):
            resz_masks[i] = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
        masks = resz_masks

        i_split = [np.array(train_idx), np.array(val_idx), np.array(test_idx)]
        return imgs, poses, [H, W, focal], masks, i_split
    
    def generate_render_poses(self, c2ws, N_views=30): 
        N = len(c2ws)
        rotvec, positions = [], []
        rotvec_inteplat, positions_inteplat = [], []
        weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
        for i in range(N):
            r = R.from_matrix(c2ws[i, :3, :3])
            euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
            if i:
                mask = np.abs(euler_ange - rotvec[0])>180
                euler_ange[mask] += 360.0
            rotvec.append(euler_ange)
            positions.append(c2ws[i, :3, 3:].reshape(1, 3))
            if i:
                rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
                positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

        rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
        positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

        c2ws_render = []
        angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
        for rotvec, position in zip(angles_inteplat, positions_inteplat):
            c2w = np.eye(4)
            c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
            c2w[:3, 3:] = position.reshape(3, 1)
            c2ws_render.append(c2w.copy())
        c2ws_render = np.stack(c2ws_render)
        return c2ws_render
    
class DTUDatasetSPARF(DTUDatasetBase): 
    def __init__(self, basedir, name, 
        n_train_imgs, 
        split_type="pixelnerf",
        no_test_rest=True, 
        mask_path=None, 
        voxel_num=64, 
        radius_div=4, 
        scene_pad=0, 
        chunk=32*1024, 
        n_sur_pts=9, 
        n_nxt_ray_pts=9, 
        pre_save_dir="vox_ray_storage/", 
        eval=False):

        self.dataset = "DTU_{}".format(name)
        self.device = torch.device("cuda")

        assert n_train_imgs in [3, 6, 9]
        self.n_train_imgs = n_train_imgs
        self.n_sur_pts = n_sur_pts
        self.n_nxt_ray_pts = n_nxt_ray_pts
        self.voxel_num = voxel_num
        self.radius_div = radius_div
        
        self.imgs, self.poses, self.poses_w2c, self.intrinsics, self.render_poses, self.hwf, self.test_masks, self.i_split = \
            self.load_dtu_data(basedir, mask_path, split_type, n_train_imgs, no_test_rest)
        # [49, 300, 400, 3]
        print('Loaded DTU SPARF ', self.imgs.shape, self.poses.shape, self.render_poses.shape, self.hwf, basedir)
        
        self.i_train, self.i_val, self.i_test = self.i_split
        self.near, self.far = 1.2, 5.2 # SPARF config

        H, W, focal = self.hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]])

        rays_o_all, rays_d_all = self.get_all_rays(H, W, self.intrinsics) # [B, HW, 3], include test poses
        self.rays_o_all, self.rays_d_all = rays_o_all.numpy().reshape(-1, 3), rays_d_all.numpy().reshape(-1, 3) # [B*HW, 3]
        self.scene_range = self.get_scene_range(self.near, self.far)
        self.scene_range += scene_pad
        self.rays_o = rays_o_all[self.i_train].reshape(-1, 3) # [n_train_imgs*HW, 3]
        self.rays_d = rays_d_all[self.i_train].reshape(-1, 3) # [n_train_imgs*HW, 3]
        self.rays_rgb = self.imgs[self.i_train].reshape(-1, 3) # [n_train_imgs*HW, 3]
        
        self.voxel_size = self.scene_range / self.voxel_num
        self.xyzmin, self.xyzmax = -0.5*self.scene_range, 0.5*self.scene_range

        self.save_dir = pre_save_dir+"{}_voxel_flag_{}x{}_{}vox_range{}_{}images_sparfreader".format(
            self.dataset, H, W, self.voxel_num, str(self.scene_range), self.n_train_imgs)
        self.prefix = "{}x{}_{}vox_range{}_{}images_sparfreader".format(
            H, W, self.voxel_num, str(self.scene_range), self.n_train_imgs)
        print("Save_dir: {}, prefix: {}".format(self.save_dir, self.prefix))
        self.voxel_flag, self.in_voxel_num = None, None
        if not eval: 
            if not os.path.exists(self.save_dir):
                self.voxel_flag, self.in_voxel_sum = self.get_ray_voxel_intersect(chunk)
                single_proc_write(self.save_dir, self.prefix, self.voxel_flag, self.in_voxel_sum)
            else: 
                self.in_voxel_sum = np.load(os.path.join(self.save_dir, self.prefix+"_voxel_sum.npy"))

        self.grid_sampler = GridPointsSampler(self.n_sur_pts, self.voxel_size, "split_region")

    def get_scene_range(self, near, far): 
        """a reference of how to decide scene_range, though it is hard-coded"""
        rays_o = self.rays_o_all
        rays_d = self.rays_d_all
        
        self.print_range(near, far, rays_o, rays_d)

        scene_range = 6
        return scene_range
    
    def get_loaded_info(self):
        return self.imgs, self.poses, self.poses_w2c, self.intrinsics, \
                self.render_poses, self.hwf, self.test_masks, self.i_split
    
    def get_all_rays(self, H, W, intrinsics): 
        rays_o, rays_d = camera.get_center_and_ray(
            pose_w2c=torch.from_numpy(self.poses_w2c), 
            H=H, W=W, intr=torch.from_numpy(intrinsics))
        # [B, HW, 3]
        return rays_o, rays_d
    
    def load_dtu_data(self, 
        path, mask_path=None, 
        split_type="pixelnerf", 
        n_input_views=-1, 
        no_test_rest=True): 
        """dtu loader from SPARF"""
        
        scene_path = path
        img_path = os.path.join(scene_path, "image")
        if not os.path.isdir(img_path):
            raise FileExistsError(img_path)

        # all images
        file_names = [f.split(".")[0] for f in sorted(os.listdir(img_path))]
        rgb_files = [os.path.join(img_path, f) for f in sorted(os.listdir(img_path))]
        pose_indices = [int(os.path.basename(e)[:-4]) for e in rgb_files] # this way is safer than range
        img_names = [f'{int(i.split("/")[-1].split(".")[0]):03d}.png' for i in rgb_files]
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

        imgs = []
        for fname in rgb_files: 
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
            imgs.append(image)
            
        imgs = np.stack(imgs, 0)
        
        camera_info = np.load(os.path.join(scene_path, "cameras.npz"))

        scaling_factor = 1./300. 
        intrinsics = []
        poses_c2w = []
        poses_w2c = []
        for p in pose_indices:
            P = camera_info[f"world_mat_{p}"] # Projection matrix 
            P = P[:3]  # (3x4) projection matrix
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K /= K[2, 2]  # 3x3 intrinsics matrix

            pose_c2w_ = np.eye(4, dtype=np.float32) # camera to world
            pose_c2w_[:3, :3] = R.transpose()
            pose_c2w_[:3, 3] = (t[:3] / t[3])[:, 0]

            intrinsics_ = np.eye(4)
            intrinsics_[:3, :3] = K
            scale_mat = camera_info.get(f"scale_mat_{p}")
            if scale_mat is not None:
                norm_trans = scale_mat[:3, 3:]
                pose_c2w_[:3, 3:] -= norm_trans
                # 1/300, scale the world
                norm_scale = np.diagonal(scale_mat[:3, :3])[..., None]
                # here it is 3 values, but equal to each other!
                assert norm_scale.mean() == 300.
                # I directly use this scaling factor to scale the depth
                # it is hardcoded in self.scaling_factor 
                # If this assertion doesn't hold, them self.scaling_factor should be equal to 1./norm_scale
                # Importantly, the norm_scale must be equal for all directions, otherwise that wouldn't scale
                # the depth map properly. 

            pose_c2w_[:3, 3:] *= scaling_factor

            poses_c2w.append(pose_c2w_)
            intrinsics.append(intrinsics_)
            poses_w2c.append(np.linalg.inv(pose_c2w_))

        intrinsics = np.stack(intrinsics, axis=0)
        poses_c2w = np.stack(poses_c2w, axis=0) # [49, 4, 4]
        poses_w2c = np.stack(poses_w2c, axis=0)
        
        intrinsics = intrinsics[:, :3, :3].astype(np.float32) # [49, 3, 3]
        poses_w2c = poses_w2c[:, :3].astype(np.float32) # [49, 3, 4]

        H, W = imgs[0].shape[:2]
        focal = intrinsics[0][0, 0]
        
        if split_type == "pixelnerf":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
            test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
            # train_idx = train_idx+exclude_idx
        elif split_type == "pixelnerf_reduced_testset":
            train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13, 24, 30, 41, 47, 43, 29, 45,
                    34, 33]
            test_idx = [1, 2, 9, 10, 11, 12, 14, 15, 23, 26, 27, 31, 32, 35, 42, 46]
        
        indices_train, indices_test = train_idx, test_idx
        indices_train = indices_train[:n_input_views]
        indices_val = indices_test.copy()
        i_split = [indices_train, indices_val, indices_test]

        # mask_path
        # ../data/DTU_mask/scan114
        masks = None
        if mask_path is not None:
            if "mask" in os.listdir(mask_path):
                mask_path = os.path.join(mask_path, "mask")
            test_imgnames = [img_names[i] for i in indices_test]
            maskfiles = [os.path.join(mask_path, name) for name in test_imgnames] 

        if mask_path is not None:
            masks = []
            for f in maskfiles:
                with open(f, "rb") as imgin:
                    image = np.array(Image.open(imgin), dtype=np.float32)[:, :, :3]/255.
                    image = (image == 1).astype(np.float32)
                masks.append(image)
            masks = np.stack(masks, 0)

        resz_masks = np.zeros((masks.shape[0], H, W, 3))
        for i in range(masks.shape[0]):
            resz_masks[i] = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
        masks = resz_masks

        # render poses from SPARF
        n_frame = 60
        poses_w2c_render = torch.inverse(torch.from_numpy(poses_c2w[indices_test]))[:, :3].float()
        poses_w2c_render = poses_w2c_render.cuda()
        # NOTE: wierd bug without moving it to gpu

        scale = 0.7
        test_poses_w2c = poses_w2c_render
        idx_center = (test_poses_w2c-test_poses_w2c.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
        pose_novel_2 = camera.get_novel_view_poses(None,test_poses_w2c[idx_center],N=n_frame,scale=scale)

        scale = 0.5
        test_poses_w2c = poses_w2c_render
        idx_center = (test_poses_w2c-test_poses_w2c.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
        pose_novel_1 = camera.get_novel_view_poses(None,test_poses_w2c[idx_center],N=n_frame,scale=scale)

        pose_novel_2 = torch.flip(pose_novel_2, (0, ))
        pose_novel = torch.cat((pose_novel_1, pose_novel_2), dim=0)
        render_poses = pose_novel.cpu().numpy()

        return imgs, poses_c2w, poses_w2c, intrinsics, render_poses, [H, W, focal], masks, i_split

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--voxel_num", type=int, default=64, help='number of voxels scene is splitted into')
    parser.add_argument("--scene", type=int, default=8, help='scan id of dtu dataset')
    parser.add_argument("--n_train_imgs", type=int, default=3, help='number of training views')
    parser.add_argument("--dtu_reader", type=str, help='type of dtu reader')
    
    return parser

if __name__ == "__main__": 
    parser = config_parser()
    args = parser.parse_args()
    assert args.dtu_reader in ["sparf", "default"]
    
    name = "scan{}".format(args.scene)
    DTUDataset = DTUDatasetSPARF if args.dtu_reader == "sparf" else DTUDatasetDefault
    dtu_dataset = DTUDataset(
        basedir="./data/rs_dtu_4/DTU/{}/".format(name), 
        mask_path="./data/DTU_mask/idrmasks/{}/".format(name), 
        name=name, n_train_imgs=args.n_train_imgs, 
        split_type="pixelnerf", no_test_rest=True, 
        voxel_num=args.voxel_num, radius_div=4, 
        chunk=64*1024, pre_save_dir="./vox_ray_storage/")

    print("Done with dtu scan{} preprocessing. n_train: {}. reader: {}. voxel_num: {}. ".format(
        args.scene, args.n_train_imgs, args.dtu_reader, args.voxel_num))
    
    sys.exit(0)