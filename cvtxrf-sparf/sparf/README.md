# CVT-xRF: Contrastive In-Voxel Transformer for 3D Consistent Radiance Fields from Sparse Inputs

<a href='https://arxiv.org/abs/2403.16885'><img src='https://img.shields.io/badge/arXiv-2403.16885-b31b1b.svg'></a> &nbsp; <a href='https://zhongyingji.github.io/CVT-xRF/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  

This repository integrates the proposed CVT module into [SPARF](https://github.com/google-research/sparf). The [diff.txt](./diff.txt) file contains the list of modified or newly added files compared to the original repository.

**NOTE:** for historical reasons, "spc" in the code or the following scripts refer to "CVT", which is relevant to the proposed CVT module of our paper. 

--------------------------------------
## Installation
Please follow the installation step in [SPARF](https://github.com/google-research/sparf) to set up an environment. It is strongly suggested to run the following command to check if everything is set correctly: 
```
python third_party/test_pdcnet_installation.py
```

--------------------------------------
## Datasets
For CVT-xRF (w/ SPARF) (this modified codebase), we only support DTU dataset for training and evaluation. 
Modify several lines in `source/admin/local.py` to define the path of the DTU dataset: 
```
self.dtu = 'YOUR_PATH_TO_DTU/rs_dtu_4/DTU'
self.dtu_depth = '' # it is OK to leave it null
self.dtu_mask = 'YOUR_PATH_TO_DTU_MASK/idrmasks/'
```

--------------------------------------
## Running the code
### Training
* **Preprocess:** Our proposed voxel-based ray sampling module records rays from training views that each voxel intersects with. Templates of running the preprocessing step are shown below, remember to replace `<input_views>` and `<scene>` with specific numbers of training views and scenes ([Examples](./scripts/examples_sparf_preprocess.sh)). This step typcalily takes 5-10 minutes. 
  ```
  # DTU template
  python run_trainval.py nerf_training_w_gt_poses/dtu spc_sparf \
  --scene=scan<scene> --train_sub <input_views>
  # <input_views>: {3, 6, 9}
  # <scene>: {8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114}
  ```
  **NOTE:** 
    * The preprocessing script is identical to the training script to be shown below; 
    * If the preprocessing step is finished correctly, there is a `spc_cache/` in the current directory; 
    * If you have already run our method with the baseline of BARF, i.e., [CVT-xRF (w/ BARF)](../../), then you can reuse the preprocessed results at `../../vox_ray_storage/`: 
      ```
      mkdir spc_cache/
      ln -s ../../../vox_ray_storage/DTU_scan<scene>_voxel_flag_300x400_64vox_range6_<input_views>images_sparfreader ./spc_cache/DTU_scan<scene>_voxel_flag_300x400_64vox_range6_<input_views>images    
      ```
      ([Examples](./scripts/examples_sparf_preprocess_reuse.sh))

      Otherwise, the above preprocessing step is necessary for each scene before training. 

* **Train:** 
  Once you have completed the preprocessing step, you can train the model using the provided script. Make sure to replace `<input_views>` and `<scene>` in the script with the actual values you used when running the preprocessing script ([Examples](./scripts/examples_sparf_train.sh)). 
  ```
  # DTU template
  python run_trainval.py nerf_training_w_gt_poses/dtu spc_sparf \
  --scene=scan<scene> --train_sub <input_views>
  ```
  The training lasts more than 10 hours on a 3090Ti GPU. The above scripts also evaluate on test split, and render videos at the end of the training. 

### Evaluation
Remember to replace the `<input_views>` and `<scene>` in the following scripts with the ones that you want to evaluate or render. 

* **Test a trained model:**
  ```
  # DTU template
  python eval.py --ckpt_dir checkpoints/nerf_training_w_gt_poses/dtu/subset_<input_views>/scan<scene>/spc_sparf --out_dir ./rendertest/ --expname spc_scan<scene>_<input_views>v --plot=True
  ```

--------------------------------------
## Citation
If you find our project helpful, please cite it as: 
```
@inproceedings{zhong2024cvt,
  title={CVT-xRF: Contrastive In-Voxel Transformer for 3D Consistent Radiance Fields from Sparse Inputs},
  author={Zhong, Yingji and Hong, Lanqing and Li, Zhenguo and Xu, Dan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
