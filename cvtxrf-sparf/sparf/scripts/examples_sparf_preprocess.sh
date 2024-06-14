
# NOTE: the preprocessing script is identical to the training script
# the program will check if there is a correspondent directory in spc_cache/
# If it exists, e.g., `spc_cache/DTU_scan40_voxel_flag_300x400_64vox_range6_3images/`. 
# the program will start the training process. Otherwise, it will run 
# the preprocessing step. 

# 3-view
# scene 40
python run_trainval.py nerf_training_w_gt_poses/dtu spc_sparf --scene=scan40 --train_sub 3

# 6-view
# scene 40
python run_trainval.py nerf_training_w_gt_poses/dtu spc_sparf --scene=scan40 --train_sub 6

# 9-view
# scene 40
python run_trainval.py nerf_training_w_gt_poses/dtu spc_sparf --scene=scan40 --train_sub 9

