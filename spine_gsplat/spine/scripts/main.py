
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
from enum import Enum
import hydra
from hydra.utils import instantiate
import argparse
import json
import datetime
from omegaconf.dictconfig import DictConfig

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path

from spine.encoders.image_encoder import BaseImageEncoder
from spine.encoders.dino_encoder import DINONetworkConfig, DINONetwork
from spine.encoders.vggt_encoder import VGGTNetworkConfig, VGGTNetwork
from spine.encoders.clip_encoder import CLIPNetworkConfig, CLIPNetwork

from spine.scripts.evaluator.spine_eval import SpineEvalConfig, SpineEval

def main(cfg):
    
    # ----------------------------------------------------------------------- #
    # Radiance field config paths
    # ----------------------------------------------------------------------- #

    with open(cfg.evaluation.rad_field_config_path, "r") as fh:
        # config paths
        rad_field_configs: dict = json.load(fh)

    # ----------------------------------------------------------------------- #
    # SPINE eval config
    # ----------------------------------------------------------------------- #

    # spine eval config
    spine_eval_cfg = instantiate(
        cfg.evaluators.spine_cfg
    )

    # config paths
    spine_eval_cfg.rf_config_paths = rad_field_configs

    # semantic encoders
    spine_eval_cfg.sem_encoders = cfg.semantic_encoders
    
    # option to load semantic encoders
    spine_eval_cfg.enable_semantic_encoders = cfg.evaluation.enable_semantic_encoders

    # save results
    spine_eval_cfg.save_results = cfg.evaluation.save_results
    
    # base output path
    base_ouput_path = cfg.evaluation.get("base_output_path", None)
    if base_ouput_path is not None:
        spine_eval_cfg.base_output_path = Path(cfg.evaluation.base_output_path)
    
    # ----------------------------------------------------------------------- #
    # Evaluate: Initialize
    # ----------------------------------------------------------------------- #
    spine_eval = instantiate(
        cfg.evaluators.spine,
        cfg=spine_eval_cfg,
    )
    
    # ----------------------------------------------------------------------- #
    # Cameras (poses) for evaluation
    # ----------------------------------------------------------------------- #
    rad_keys = list(spine_eval.rad_field.keys())
    eval_cam = {}
    for rad_key in rad_keys:
        eval_cam_scene = spine_eval.rad_field[rad_key]["base"].pipeline.datamanager.eval_dataset.cameras
    
        # TODO: workaround for 3D-OVS scenes (too few or no eval cameras so we use the train dataset)
        if "3d_ovs" in f"{rad_key}".lower():
            eval_cam_scene = spine_eval.rad_field[rad_key]["base"].pipeline.datamanager.train_dataset.cameras
        
        if len(eval_cam_scene) < cfg.evaluation.num_eval_cams:
            # number of interpolation steps for the camera
            num_cam_interp_steps = max(1, np.ceil(cfg.evaluation.num_eval_cams / (len(eval_cam_scene) - 1)).astype(int))

            # interpolate the cameras
            eval_cam_scene = get_interpolated_camera_path(
                cameras=eval_cam_scene,
                steps=num_cam_interp_steps,
                order_poses=False,
            )
        
        # select the evaluation cameras
        eval_cam[rad_key] = eval_cam_scene[:cfg.evaluation.num_eval_cams]
        
    # ----------------------------------------------------------------------- #
    # Evaluate: RF Inversion
    # ----------------------------------------------------------------------- #
    
    rf_inversion_results = spine_eval.evaluate_rf_inversion(
        cameras=eval_cam,
    )
    
    # ----------------------------------------------------------------------- #
    # Evaluate: PCA
    # ----------------------------------------------------------------------- #
    rf_pca_results = spine_eval.evaluate_semantics_pca(cameras=eval_cam)
    
    # ----------------------------------------------------------------------- #
    # Evaluate: Semantic Segmentation
    # ----------------------------------------------------------------------- #
    rf_sem_seg_results = spine_eval.evaluate_semantic_segmentation(
        cameras=eval_cam,
        object_queries=cfg.object_queries,
    )
    
    # all results
    results = {
        "rf_inversion": rf_inversion_results,
        "rf_semantics_pca": rf_pca_results,
        "rf_sem_seg_results": rf_sem_seg_results,
    }
    
    return results
 

if __name__ == "__main__":
    
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str, 
        required=False,
        default="eval_rf",
        help="Name of the config file"
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # load the configs from file
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"configs", version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)
    
    # run the main function
    main(
        cfg=cfg
    )