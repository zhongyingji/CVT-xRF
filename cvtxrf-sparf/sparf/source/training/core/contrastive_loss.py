import numpy as np
import torch
import torch.nn as nn
from source.training.core.base_losses import BaseLoss
from easydict import EasyDict as edict

from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class VoxelContrastiveLoss(BaseLoss): 
    def __init__(self, opt: Dict[str, Any], device: torch.device): 
        super().__init__(device)
        
        self.opt = opt
        
        self.n_pos = opt.voxel_sampler.n_rays_per_voxel
        self.sample_n_voxels = opt.voxel_sampler.sample_n_voxels
        self.T = opt.contrast_T
        self.ce = nn.CrossEntropyLoss().to(device)
        
        self.proj = nn.Sequential(
            nn.Linear(256, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
        ).to(device)

        self.label = self.__get_pseudo_label().to(device)

        self.device = device

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Output dict from the renderer. Contains important fields
                                - idx_img_rendered: idx of the images rendered (B), useful 
                                in case you only did rendering of a subset
                                - ray_idx: idx of the rays rendered, either (B, N) or (N)
                                - rgb: rendered rgb at rays, shape (B, N, 3)
                                - depth: rendered depth at rays, shape (B, N, 1)
                                - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                                - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """

        if mode != 'train':
            # only during training
            return {}, {}, {}
            
        if "spc_loader" in output_dict: 
            # rays from random sampler + dataloader
            emb = output_dict["spc_loader"]["surround_embed_fine"]
        elif "surround_embed_fine" in output_dict: 
            # only rays from dataloader
            emb = output_dict["surround_embed_fine"]
        else: 
            raise Exception("surround_embed should be in the output_dict.")
        
        loss_dict = edict(contrast=torch.tensor(0., requires_grad=True).to(self.device))

        label = self.label
        emb = self.proj(emb)
        cos_dist = cosine_dist(emb, emb) # (N, N)
        N = cos_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        # NOTE: the mat_dist is the cosine distance, not similarity

        # get a matrix, with same label as 0, different labels as 1, 
        # and set the hard positive as 1; 
        # then indexing
        rev_mat_sim = 1.0 - mat_sim
        hard_p, _, hard_p_indice, _ = _batch_hard(cos_dist, mat_sim, indice=True)
        rev_mat_sim[torch.arange(N), hard_p_indice] = 1.0
        # except hard positive, and all negative; all other entries are set with 0

        hard_p_indice_ = hard_p_indice // self.n_pos

        col_indices = rev_mat_sim.nonzero()[:, 1]
        assert col_indices.size(0) == N*(N-self.n_pos+1)
        col_indices = col_indices.reshape(N, N-self.n_pos+1)

        cos_dist_hardp = torch.gather(cos_dist, 1, col_indices) # (N, N-self.n_pos+1)
        cos_dist_hardp = (cos_dist_hardp - 1) * (-1)
        cos_dist_hardp /= self.T

        target = hard_p_indice_ * self.n_pos # the new indices for the hard positive in cos_dist_hardp
        loss = self.ce(cos_dist_hardp, target)
        loss_dict.contrast = loss
        return loss_dict, {}, {}
        
    
    def __get_pseudo_label(self): 
        pseudo_label = torch.arange(self.sample_n_voxels)[:, None] # [sample_n_voxels, 1]
        pseudo_label = pseudo_label.expand(self.sample_n_voxels, self.n_pos)
        pseudo_label = pseudo_label.contiguous().view(-1) # [batch_size, ]
        return pseudo_label