import os, time
import json
import imageio
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.append("./")
sys.path.append("../")
from dataloader.utils.ray_voxel_intersection import ray_voxel_intersect
from dataloader.sampler import GridPointsSampler
from run_nerf_helpers import get_rays_np
from dataloader.common import VoxelStorageDS, single_proc_write

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class BlenderDatasetDefault(Dataset):
    def __init__(self, basedir, name, 
        n_train_imgs, 
        half_res=False, 
        testskip=1, 
        white_bkgd=False, 
        bound=[2., 6.], 
        voxel_num=64, 
        radius_div=2, 
        scene_pad=0, 
        chunk=32*1024, 
        n_sur_pts=9, 
        n_nxt_ray_pts=9, 
        train_test_split_file=None, 
        pre_save_dir="vox_ray_storage/", 
        eval=False):

        self.dataset = "Blender_{}".format(name)
        self.device = torch.device("cuda")

        assert n_train_imgs in [3, 8]
        self.n_train_imgs = n_train_imgs
        self.n_sur_pts = n_sur_pts
        self.n_nxt_ray_pts = n_nxt_ray_pts
        self.voxel_num = voxel_num
        self.radius_div = radius_div

        self.imgs, self.poses, self.render_poses, self.hwf, self.i_split = \
            self.load_blender_data(basedir, half_res, n_train_imgs, testskip, train_test_split_file)
        print("Loaded Blender ", self.imgs.shape, self.poses.shape, self.render_poses.shape, self.hwf, basedir)

        if white_bkgd:
            self.imgs = self.imgs[...,:3]*self.imgs[...,-1:] + (1.-self.imgs[...,-1:])
        else:
            self.imgs = self.imgs[...,:3]
        self.i_train, self.i_val, self.i_test = self.i_split
        self.near, self.far = bound

        H, W, focal = self.hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]
        self.K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        
        self.rays_o, self.rays_d, self.scene_range = self.get_scene_range(self.near, self.far)
        self.scene_range += scene_pad

        select_train_imgs = np.stack(
            [self.imgs[self.i_train[i]] for i in range(self.n_train_imgs)], 0)
        self.rays_rgb = select_train_imgs.reshape(-1, 3)

        
        self.voxel_size = self.scene_range / self.voxel_num
        self.xyzmin, self.xyzmax = -0.5*self.scene_range, 0.5*self.scene_range

        self.save_dir = pre_save_dir+"{}_voxel_flag_{}x{}_{}vox_range{}_{}images".format(
            self.dataset, H, W, voxel_num, str(self.scene_range), n_train_imgs)
        self.prefix = "{}x{}_{}vox_range{}_{}images".format(
            H, W, voxel_num, str(self.scene_range), n_train_imgs)
        print("Save_dir: {}, prefix: {}".format(self.save_dir, self.prefix))
        self.voxel_flag, self.in_voxel_num = None, None
        if not eval: 
            if not os.path.exists(self.save_dir):
                self.voxel_flag, self.in_voxel_sum = self.get_ray_voxel_intersect(chunk)
                single_proc_write(self.save_dir, self.prefix, self.voxel_flag, self.in_voxel_sum)
            else: 
                self.in_voxel_sum = np.load(os.path.join(self.save_dir, self.prefix+"_voxel_sum.npy"))
            
        self.grid_sampler = GridPointsSampler(self.n_sur_pts, self.voxel_size, "split_region")

    def load_blender_data(self, 
        basedir, half_res=False, 
        n_input_views=3, testskip=1, 
        train_test_split_file=None):

        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        load_blender_data_v = self.load_blender_data_3v if n_input_views == 3 else self.load_blender_data_8v
        i_split, imgs, poses = load_blender_data_v(basedir, metas, splits, testskip, train_test_split_file)
        print("i_split: {}, imgs_shape: {}, poses_shape: {}".format(i_split, imgs.shape, poses.shape))
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(metas['test']['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
        
        return imgs, poses, render_poses, [H, W, focal], i_split
    
    def load_blender_data_3v(self, basedir, metas, splits, testskip, train_test_split_file):
        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas['train']
            imgs = []
            poses = []
            
            if s == 'train':
                name = os.path.basename(basedir)
                img_idx = torch.load(train_test_split_file)[f'{name}_{s}'][:self.n_train_imgs]
            else:
                img_idx = torch.load(train_test_split_file)[f'{name}_val']

            print(s, ': ', img_idx)

            for idx in img_idx:
                frame = meta['frames'][idx]
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        return i_split, imgs, poses
        
    def load_blender_data_8v(self, basedir, metas, splits, testskip, train_test_split_file=None): 
        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []

            if s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
            
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        
        i_split[0] = [26, 86, 2, 55, 75, 93, 16, 73] # train_idx is hard-coded
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        return i_split, imgs, poses
        
    
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
        rays = np.stack([get_rays_np(self.hwf[0], self.hwf[1], self.K, p) \
                            for p in self.poses[:,:3,:4]], 0)
        # [N, ro+rd, H, W, 3]
        # use all images to get the bound, 
        # but only return n_train imgs for training.
        rays_o, rays_d = (rays[:, 0]).reshape(-1, 3), (rays[:, 1].reshape(-1, 3))
        
        self.print_range(near, far, rays_o, rays_d)
        
        scene_range = 6 if self.n_train_imgs == 3 else 4

        rays_o = np.concatenate(
            [rays_o[(self.i_train[i]*self.hwf[0]*self.hwf[1]):((self.i_train[i]+1)*self.hwf[0]*self.hwf[1])] for i in range(self.n_train_imgs)], 0)
        rays_d = np.concatenate(
            [rays_d[(self.i_train[i]*self.hwf[0]*self.hwf[1]):((self.i_train[i]+1)*self.hwf[0]*self.hwf[1])] for i in range(self.n_train_imgs)], 0)

        return rays_o, rays_d, scene_range

    def get_loaded_info(self):
        return self.imgs, self.poses, self.render_poses, self.hwf, self.i_split
    
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
        sur_pts = self.grid_sampler.sphere_sample(sur_pts, self.voxel_size/2, self.n_sur_pts)
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
        """return a ray, and surrounding points of it"""
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

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--voxel_num", type=int, default=64, help='number of voxels scene is splitted into')
    parser.add_argument("--scene", type=str, default="lego", help='scene of blender dataset')
    parser.add_argument("--n_train_imgs", type=int, default=3, help='number of training views')
    
    return parser

if __name__ == "__main__": 
    parser = config_parser()
    args = parser.parse_args()
    
    name = args.scene
    n_train_imgs = args.n_train_imgs
    voxel_num = args.voxel_num
    blender_dataset = BlenderDatasetDefault(
        basedir="./data/nerf_synthetic/{}".format(name), 
        name=name, n_train_imgs=n_train_imgs, 
        half_res=True if n_train_imgs==8 else False, 
        testskip=8, chunk=64*1024, voxel_num=voxel_num, 
        train_test_split_file="./configs/pairs.th", 
        pre_save_dir="./vox_ray_storage/")
    
    print("Done with blender preprocessing. n_train: {}. voxel_num: {}. ".format(
        args.scene, args.n_train_imgs, args.voxel_num))
    