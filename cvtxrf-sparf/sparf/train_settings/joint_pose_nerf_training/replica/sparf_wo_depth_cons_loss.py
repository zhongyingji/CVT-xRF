"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import time
from pathlib import Path
from easydict import EasyDict as edict
import os

from source.utils.config_utils import override_options
from train_settings.default_config import get_joint_pose_nerf_default_config_360_data


def get_config():
    default_config = get_joint_pose_nerf_default_config_360_data()
    
    settings_model = edict()

    # camera options    
    settings_model.camera = edict()
    settings_model.camera.initial_pose = 'sfm_pdcnet'

    # scheduling of 2 stage training
    settings_model.first_joint_pose_nerf_then_nerf = True
    settings_model.ratio_end_joint_nerf_pose_refinement = 0.25
    settings_model.barf_c2f =  [0.4,0.7] 

    # dataset
    settings_model.dataset = 'replica'
    settings_model.resize = [340, 600]
    
    settings_model.nerf = edict()                                                       # NeRF-specific options
    settings_model.nerf.depth = edict()                                                  # depth-related options
    settings_model.nerf.depth.param = 'metric'                                       # depth parametrization (for sampling along the ray)
    settings_model.nerf.fine_sampling = True                               # hierarchical sampling with another NeRF
    settings_model.nerf.ratio_start_fine_sampling_at_x = settings_model.ratio_end_joint_nerf_pose_refinement

    # flow stuff
    settings_model.use_flow = True
    settings_model.flow_backbone='PDCNet' 
    settings_model.filter_corr_w_cc = True
    # leads to slightly better results to have additional filtering on the correspondences here

    # loss type
    settings_model.loss_type = 'photometric_and_corres'
    settings_model.matching_pair_generation = 'all_to_all'

    settings_model.gradually_decrease_corres_weight = True
    settings_model.ratio_start_decrease_corres_weight = settings_model.ratio_end_joint_nerf_pose_refinement
    settings_model.corres_weight_reduct_at_x_iter = 10000

    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0.    
    settings_model.loss_weight.corres = -3. # for 10^
    return override_options(default_config, settings_model)

