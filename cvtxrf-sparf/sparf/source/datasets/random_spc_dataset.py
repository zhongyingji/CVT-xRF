import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from source.datasets.base_spc_dataset import BaseSPCDataset

class RandomSPCDataset(BaseSPCDataset): 

    def __init__(self, opt): 
        super().__init__(opt)
        self.opt = opt
    
    def __getitem__(self, idx): 
        ray_o = np.random.rand(3).astype(np.float32)
        ray_d = np.random.rand(3).astype(np.float32)
        ray_rgb = np.random.rand(3).astype(np.float32)
        
        ref_pt = np.random.rand(3).astype(np.float32)
        t_ref_pt = np.random.rand(1).astype(np.float32)
        
        sur_pts = np.random.rand(3, self.opt.voxel_sampler.n_sur_pts).astype(np.float32)
        nxt_ray_pts = np.random.rand(3, self.opt.voxel_sampler.n_nxt_ray_pts).astype(np.float32)
        nxt_ray_t_pts = np.random.rand(self.opt.voxel_sampler.n_nxt_ray_pts).astype(np.float32)

        vox_end = np.array([0.]).astype(np.float32)

        return ray_o, ray_d, ray_rgb, ref_pt, t_ref_pt, \
                np.concatenate([sur_pts, nxt_ray_pts], axis=1), \
                nxt_ray_t_pts, vox_end
        
        
    
    def __len__(self): 
        return np.iinfo(np.int64).max
        # NOTE: dirty hack
        
