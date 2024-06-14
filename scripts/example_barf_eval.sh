# DTU
# scan40

# 3-view
python run_nerf.py --config configs/barf/dtu/3v.txt \
--expname dtu_scan40_n3_barf \
--datadir ./data/rs_dtu_4/DTU/scan40 \
--dtu_maskdir ./data/DTU_mask/idrmasks/scan40 \
--test_only

# 6-view
python run_nerf.py --config configs/barf/dtu/6v.txt \
--expname dtu_scan40_n6_barf \
--datadir ./data/rs_dtu_4/DTU/scan40 \
--dtu_maskdir ./data/DTU_mask/idrmasks/scan40 \
--test_only

# 9-view
python run_nerf.py --config configs/barf/dtu/9v.txt \
--expname dtu_scan40_n9_barf \
--datadir ./data/rs_dtu_4/DTU/scan40 \
--dtu_maskdir ./data/DTU_mask/idrmasks/scan40 \
--test_only

# --------------------------------------------------------

# Synthetic
# lego

# 3-view
python run_nerf.py --config configs/barf/blender/3v.txt \
--expname blender_lego_n3_barf \
--datadir ./data/nerf_synthetic/lego \
--test_only

# 8-view
python run_nerf.py --config configs/barf/blender/8v.txt \
--expname blender_lego_n8_barf \
--datadir ./data/nerf_synthetic/lego \
--test_only


# --------------------------------------------------------

# To render a video from a trained model, replace --test_only with --render_only 
# and run the script agian

# DTU, scan40, 3-view
python run_nerf.py --config configs/barf/dtu/3v.txt \
--expname dtu_scan40_n3_barf \
--datadir ./data/rs_dtu_4/DTU/scan40 \
--dtu_maskdir ./data/DTU_mask/idrmasks/scan40 \
--render_only

