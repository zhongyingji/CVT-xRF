import numpy as np
import os, imageio
from pathlib import Path
import cv2
import torch
import sys
from torch.utils.data.sampler import BatchSampler

class VoxelSampler(BatchSampler):
    def __init__(self, voxel_flag, in_voxel_sum, voxel_num, batch_size, 
        sample_n_voxels=256, 
        n_rays_per_voxel=4, 
        precrop_iters=500, 
        n_rays_thsh=8, 
        weighted_sampling=False):
        self.voxel_flag = voxel_flag # (voxel_num**3, set)
        self.in_voxel_sum = in_voxel_sum
        self.voxel_num = voxel_num
        self.batch_size = batch_size
        self.sample_n_voxels = sample_n_voxels
        self.n_rays_per_voxel = n_rays_per_voxel
        self.n_rays_thsh = n_rays_thsh # NOTE: only the voxels with more than this number of rays are sampled

        assert self.batch_size == self.sample_n_voxels * self.n_rays_per_voxel
    
        # self.sorted_voxel_idx = np.argsort(self.in_voxel_sum)[::-1] # sort voxel with the number of rays intersect with it
        self.voxel_cursor = np.zeros(voxel_num**3, dtype=int) # start point
        self.idx_pool = [np.arange(j) for j in self.in_voxel_sum]

        # self.valid_voxel_pool = np.ones(voxel_num**3)
        # self.valid_voxel_pool[np.where(in_voxel_sum<self.n_rays_thsh)[0]] = 0

        # TODO: check whether it is re-initialized after all samples are done
        
        self.weighted_sampling = weighted_sampling
        if not weighted_sampling: 
            self.sample_voxel_fn = self.sample_valid_voxel
        else: 
            self.sample_voxel_fn = self.weighted_sample_valid_voxel

        self.precrop_iters = precrop_iters
        self.reset()


    def reset(self):
        self.voxel_cursor *= 0
        
        self.valid_voxel_pool = np.ones(self.voxel_num**3)
        self.valid_voxel_pool[np.where(self.in_voxel_sum<self.n_rays_thsh)[0]] = 0
        
        if self.weighted_sampling: 
            # For those voxels with over 100 rays, their sampling weight should be 5 times over those with less rays
            mask_in_voxel_sum = (self.in_voxel_sum >= 100).astype(np.float32)
            mask_in_voxel_sum *= 0.5
            mask_in_voxel_sum[mask_in_voxel_sum==0] = 0.1
            mask_in_voxel_sum[np.where(self.in_voxel_sum<self.n_rays_thsh)[0]] = 0.
            
            self.voxel_sample_weights = mask_in_voxel_sum/mask_in_voxel_sum.sum()
            print("voxel_sample_weights: ", self.voxel_sample_weights)
            
        self.cnt = 0
        for idp in self.idx_pool:
            np.random.shuffle(idp)
    

    def __iter__(self): 
        while True: 
            ret = []
            voxel_indices, _ = self.sample_voxel_fn()
            for vox_idx in voxel_indices: 
                ret_per_vox = []
                if self.in_voxel_sum[vox_idx] < self.n_rays_per_voxel: 
                    ray_indices = self.idx_pool[vox_idx][:]
                    # print("less: ", ray_indices)
                else: 
                    st_cursor = np.random.randint(0, self.in_voxel_sum[vox_idx]-self.n_rays_per_voxel+1)
                    ray_indices = self.idx_pool[vox_idx][st_cursor:(st_cursor+self.n_rays_per_voxel)]
                    # print("more. total_rays_in_voxel: {}, cursor: {}, ray_indices: {}".format(self.in_voxel_sum[vox_idx], st_cursor, ray_indices))
                ray_indices = list(ray_indices)
                
                if len(ray_indices) < self.n_rays_per_voxel: 
                    ray_indices.extend([ray_indices[-1] for _ in range(self.n_rays_per_voxel-len(ray_indices))])
                
                for j in range(self.n_rays_per_voxel): 
                    ret_per_vox.append([vox_idx, ray_indices[j], ray_indices[(j+1)%self.n_rays_per_voxel], j==(self.n_rays_per_voxel-1)])
                    # voxel_idx, ray index in voxel, next ray index in the same voxel, if it is ended of the voxel
                
                ret.extend(ret_per_vox)
            
            self.cnt += 1
            yield ret

            if self.cnt % 20000 == 0: 
                self.reset()
            
            
    def __len__(self):
        return np.sum(self.in_voxel_sum) // self.batch_size

    
    def sample_valid_voxel(self):
        # return sample_n_voxels indices
        valid = np.where(self.valid_voxel_pool>0)[0]
        return np.random.choice(valid, self.sample_n_voxels, replace=False), False
    
    def weighted_sample_valid_voxel(self): 
        return np.random.choice(self.voxel_num**3, self.sample_n_voxels, replace=False, p=self.voxel_sample_weights), False


class GridPointsSampler():
    def __init__(self, n_points, grid_size, sampling_scheme="split_region"):
        self.n_points = n_points
        # sample 2*n_points in total
        self.grid_size = grid_size

        if sampling_scheme == "split_region":
            self.sample_fn = self.split_region_sample
            self.n_split = 2
            self.radius = self.grid_size / 2
            
        elif sampling_scheme == "random":
            self.sample_fn = self.random_sample
            self.radius = self.grid_size / 2
        
    def sphere_sample(self, center, radius, n_points):
        # center: (2, 3)
        # n_points: for each center
        vec = np.random.randn(2, n_points, 3)
        norm = np.linalg.norm(vec, axis=-1)[..., None]
        vec /= norm # (2, n_points, 3)
        r = np.random.rand(2, n_points, 1) * radius
        points = vec * r
        points = center[:, None, :] + points
        return points
        
    def split_region_sample(self, grid_x1, grid_x2, grid_y1, grid_y2, grid_z1, grid_z2):
        n_regions = self.n_split ** 3
        x1 = grid_x1 + (grid_x2-grid_x1) / 4
        x2 = (grid_x1+grid_x2)/2 + (grid_x2-grid_x1)/4
        y1 = grid_y1 + (grid_y2-grid_y1) / 4
        y2 = (grid_y1+grid_y2)/2 + (grid_y2-grid_y1)/4
        z1 = grid_z1 + (grid_z2-grid_z1) / 4
        z2 = (grid_z1+grid_z2)/2 + (grid_z2-grid_z1)/4
        
        center_cand = np.stack(
            [[x1, y1, z1], 
            [x1, y1, z2], 
            [x1, y2, z1], 
            [x1, y2, z2], 
            [x2, y1, z1], 
            [x2, y1, z2], 
            [x2, y2, z1], 
            [x2, y2, z2]
            ], axis=0
        ) # (8, 3)
        # idx1 = np.random.randint(8)
        # idx2 = np.random.randint(8)
        indices = np.random.choice(8, 2, replace=False)
        idx1, idx2 = indices[0], indices[1]
        center = center_cand[[idx1, idx2]] # (2, 3)
        points = self.sphere_sample(center, self.radius, self.n_points)
        return points
        
    def random_sample(self, grid_x1, grid_x2, grid_y1, grid_y2, grid_z1, grid_z2): 
        rand = np.random.rand(2*3).reshape(2, 3)
        centerx = grid_x1 + (grid_x2-grid_x1)*rand[:, [0]] # (2, 1)
        centery = grid_y1 + (grid_y2-grid_y1)*rand[:, [1]]
        centerz = grid_z1 + (grid_z2-grid_z1)*rand[:, [2]]
        center = np.concatenate([centerx, centery, centerz], axis=1).astype(np.float32) # (2, 3)
        
        points = self.sphere_sample(center, self.radius, self.n_points)
        return points # (2, n_points, 3)

    def sample(self, grid_x1, grid_x2, grid_y1, grid_y2, grid_z1, grid_z2):
        return self.sample_fn(grid_x1, grid_x2, grid_y1, grid_y2, grid_z1, grid_z2)


class VoxelStorageDS():
    def __init__(self, max_len=1e5):
        self.pool = []

    def add(self, ray_idx, 
        in_coordx, in_coordy, in_coordz, 
        out_coordx, out_coordy, out_coordz, 
        ts_in, ts_out, r, g, b):
        self.pool.append([ray_idx, in_coordx, in_coordy, in_coordz, 
        out_coordx, out_coordy, out_coordz, 
        ts_in, ts_out, r, g, b])

    def get_n_rays(self):
        return len(self.pool)
    
    def get_info(self):
        return self.pool