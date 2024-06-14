import torch
from source.datasets.dtu_spc_dataset import DTUSPCDataset
from source.datasets.random_spc_dataset import RandomSPCDataset
from source.datasets.spc_utils import VoxelSampler

def get_spc_dtu_loader(opt, all_data, device):
    dtu_spc_dataset = DTUSPCDataset(opt, all_data, device=device)

    sampler = VoxelSampler(
        dtu_spc_dataset.voxel_flag, 
        dtu_spc_dataset.in_voxel_sum, 
        voxel_num=opt.voxel_sampler.voxel_num, 
        sample_n_voxels=opt.voxel_sampler.sample_n_voxels, 
        n_rays_per_voxel=opt.voxel_sampler.n_rays_per_voxel, 
        batch_size=opt.voxel_sampler.sample_n_voxels*opt.voxel_sampler.n_rays_per_voxel, 
        n_rays_thsh=opt.voxel_sampler.voxel_sampler_rays_thsh, 
        precrop_iters=0, 
        weighted_sampling=False, 
    )

    # torch.multiprocessing.set_start_method("spawn")
    data_loader = torch.utils.data.DataLoader(
        dtu_spc_dataset, batch_sampler=sampler, num_workers=opt.dataloader_workers, pin_memory=True)
    
    return data_loader


def get_spc_random_loader(opt, all_data): 
    random_spc_dataset = RandomSPCDataset(opt)

    data_loader = torch.utils.data.DataLoader(
        random_spc_dataset, 
        batch_size=opt.voxel_sampler.sample_n_voxels*opt.voxel_sampler.n_rays_per_voxel, 
    )

    return data_loader