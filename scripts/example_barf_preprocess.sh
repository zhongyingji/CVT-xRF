# DTU
# scan40

# 3-view
python dataloader/dtu_loader.py --n_train_imgs 3 \
--scene 40 --dtu_reader "sparf" 

# 6-view
python dataloader/dtu_loader.py --n_train_imgs 6 \
--scene 40 --dtu_reader "sparf" 

# 9-view
python dataloader/dtu_loader.py --n_train_imgs 9 \
--scene 40 --dtu_reader "sparf" 


# --------------------------------------------------------

# Synthetic
# lego

# 3-view
python dataloader/blender_loader.py --n_train_imgs 3 \
--scene "lego"

# 8-view
python dataloader/blender_loader.py --n_train_imgs 8 \
--scene "lego"