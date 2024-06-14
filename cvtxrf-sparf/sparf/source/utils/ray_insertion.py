import numpy as np
import torch
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict

def insert_pt_to_ray(ray_rgb_alph: torch.Tensor, ray_zvals: torch.Tensor, insert_pts: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:  
        ray_rgb_alph: [num_rays, num_samples along ray, 4]. 
        ray_zvals: [num_rays, num_samples along ray]. 
        ---------------------------------------------
        dict of insert_pts: 
        rgb_ref: [num_rays, 1, 3]. The reference point of each ray.
        rgb_nxt_ray: [num_rays, n_nxt_ray_pts, 3]. 
        alph_ref: [num_rays, 1, 1]. The predicted alpha of each reference point. 
        alph_nxt_ray: [num_rays, n_nxt_ray_pts, 1].
        ts_ref: [num_rays, 1]. The t of each reference point. 
        ts_nxt_ray: [num_rays, n_nxt_ray_pts]. 
        vox_end: [num_rays, 1]
        ---------------------------------------------
        for a batch of rays, the first ray can only use its reference point to insert to the ray; 
        the other rays can use their reference points and the points from transformer of previous indices.

        # insert_rgb: [num_rays, 3]. 
        # insert_alph: [num_rays, 1]. 
        # insert_ts: [num_rays, 1]. 
    """
    
    vox_end = insert_pts["vox_end"]
    split_idx = torch.where(vox_end==1.)[0]
    # print("Check the split_idx in `insert_pt_to_ray`. ", split_idx)
    insert_alph, insert_ts = [], []
    tmp_cursor = 0

    insert_alph = torch.cat([insert_pts["alph_ref"], insert_pts["alph_nxt_ray"]], dim=1)
    insert_ts = torch.cat([insert_pts["ts_ref"], insert_pts["ts_nxt_ray"]], dim=1)
    insert_rgb = torch.cat([insert_pts["rgb_ref"], insert_pts["rgb_nxt_ray"]], dim=1)
    # (num_rays, 1+n_nxt_ray_pts, 3)

    
    """
    for i in range(insert_rgb.shape[1]):
        res, res_zvals = one_insertion(res, res_zvals, insert_rgb[:, i], insert_alph[:, i], insert_ts[:, [i]])
        # print(i, res.shape, res_zvals.shape)
    """
    
    
    """
    res, res_zvals = multi_insertion(ray_rgb_alph, ray_zvals, insert_rgb, insert_alph, insert_ts)
    misc = {
        "transformer_output_alpha": F.relu(insert_alph), 
    }
    return res, res_zvals, ray_zvals, misc
    """

    inserted_ray_rgb_alph, inserted_zvals = multi_insertion(ray_rgb_alph, ray_zvals, insert_rgb, insert_alph, insert_ts)

    return {
        "inserted_ray_rgb_alph": inserted_ray_rgb_alph, # (num_rays, num_samples+1+n_nxt_ray_pts, 4)
        "inserted_zvals": inserted_zvals, # (num_rays, num_samples+1+n_nxt_ray_pts)
        "transformer_output_alpha": insert_alph, 
    }

    

def multi_insertion(ray_rgb_alph: torch.Tensor, 
                    ray_zvals: torch.Tensor, insert_rgb: torch.Tensor, 
                    insert_alph: torch.Tensor, insert_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: 
    # concat to the end and sort w.r.t. insert_ts
    """Args: 
        ray_rgb_alph: [num_rays, num_samples along ray, 4]. 
        ray_zvals: [num_rays, num_samples along ray]. 
        insert_rgb: [num_rays, 1+n_nxt_ray_pts, 3]. 
        insert_alph: [num_rays, 1+n_nxt_ray_pts, 1]. 
        insert_ts: [num_rays, 1+n_nxt_ray_pts]. 
    """
    insert_rgb_alph = torch.cat([insert_rgb, insert_alph], dim=-1) # [num_rays, 1+n_nxt_ray_pts, 4]
    
    ray_rgb_alph = torch.cat([ray_rgb_alph, insert_rgb_alph], dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts, 4]
    ray_zvals = torch.cat([ray_zvals, insert_ts], dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts]
    
    nrays, nsamples, ndim = ray_rgb_alph.shape

    res_zvals, sort_idx = torch.sort(ray_zvals, dim=1) # [num_rays, num_samples+1+n_nxt_ray_pts]
    
    # TODO: how to index the ray_rgb_alph according to sort_idx
    res = torch.gather(ray_rgb_alph, 1, sort_idx[..., None].expand(nrays, nsamples, ndim))

    return res, res_zvals
