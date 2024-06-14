import os
import numpy as np

"""data structure for each voxel
storing rays that intersect with it"""
class VoxelStorageDS():
    def __init__(self):
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

def single_proc_write(save_dir, prefix, voxel_flag, voxel_sum):
    cpu_num = 1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, prefix+"_voxel_sum"), voxel_sum)
    print("cpu_num: {}, prefix: {}, save_dir: {}".format(
        cpu_num, prefix, save_dir
    ))
    for ridx in range(len(voxel_flag)):
        record = voxel_flag[ridx]
        record = np.array(record.get_info()) # list
        save_file = prefix + "_ray_{}".format(ridx)
        np.save(os.path.join(save_dir, save_file), record)
        print("ridx: {}. {}".format(ridx, os.path.join(save_dir, save_file)))