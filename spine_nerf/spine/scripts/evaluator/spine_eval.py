import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Callable

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
from enum import Enum
import gc
import random
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision.ops import box_convert
from hydra.utils import instantiate

# metrics
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from spine.encoders.image_encoder import BaseImageEncoder
from spine.encoders.dino_encoder import DINONetworkConfig, DINONetwork
from spine.encoders.vggt_encoder import VGGTNetworkConfig, VGGTNetwork
from spine.encoders.clip_encoder import CLIPNetworkConfig, CLIPNetwork

from spine.scripts.evaluator.base_eval import RadianceFieldEvalConfig, RadianceFieldEval
from spine.scripts.utils.radiance_field_utils import RFModel, load_dataset, SH2RGB
from spine.scripts.utils.pca_utils import sobel_gradients
from spine.scripts.utils.inversion_utils import SE3error
# from spine.spine import _quat_to_rotmat
from spine.spine_utils import _lie_algebra_to_rotmat
from spine.scripts.utils.segmentation_utils import (
    load_gdino, get_gdino_config, predict, annotate,  # GroundingDINO
    load_sam2,  # SAM2
    compute_mIoU,
    compute_segmentation_accuracy,
    show_masks,
    plot_bbox,
    post_process_bbox,
    apply_colormap,  # colormap
    ColormapOptions,
)
from spine.scripts.utils.estimator_utils import (
    POI_Detector, 
    vec_to_rot_matrix, 
    execute_PnP_RANSAC,
    execute_iNeRF,
)


class SpineEvalConfig(RadianceFieldEvalConfig):
    # semantic encoders
    sem_encoders: dict[str, dict] = {}
    
    # option to load the semantic encoders
    enable_semantic_encoders: bool = False

    # number of PCA components
    num_pca_components: int = 4  # number of PCA components
    
    # option to visualize results
    visualize_results: bool = False
    
    # option to save results
    save_results: bool = False
    
    # enable outlier rejection in PCA
    enable_outlier_rejection_in_pca: bool = False
    
    # option to process geometry in PCA
    process_pca_geometry: bool = True
    
    # threshold for Sobel gradients in edge extraction
    gradient_sobel_norm_threshold: list[float] = [1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1
                                                  ] # can sweep over a list of values
    
    # negative queries for semantic segmentation
    negative_queries: str = "object, stuff"
    
    # option to filter segmentation masks by maximum detection score
    filter_by_max_score: bool = True

    # GroundingDINO eval config
    gdino_config: dict = get_gdino_config()
    
    # threshold for generating the predicted mask (and quantiles)
    pred_mask_threshold: float = 0.5   # value between 0 and 1
    pred_mask_quantile_q: float = 0.9  # value between 0 and 1
    
    # option to run fine inversion with the proposed method
    run_fine_inversion: bool = True
    
    # set the fine inversion method
    rf_fine_inversion_method: Callable = execute_iNeRF
    
    # set the baseline method
    rf_inversion_baseline_method: Callable = execute_iNeRF
    
    # option to initialize the baselines with the ground-truth pos
    init_with_perturbed_gt_pose: bool = True
    
    # parameters with which to perturb the ground-truth to get an initial guess for the RF inversion BASELINES
    perturb_init_guess_params = {
        "rotation": [np.deg2rad(5), np.deg2rad(30), np.deg2rad(100)],
        "translation": [0.1, 0.5, 1],
    }
    
    # other parameters for the RF inversion baseline
    rf_inversion_baseline_learning_rate: float = 1e-2
    rf_inversion_baseline_conv_threshold: float = 1e-3
    rf_inversion_baseline_max_num_iterations: int = 300
    rf_inversion_baseline_batch_size: int = 512
        
    # option to enable verbose print
    verbose_print: bool = True
    
    # base output path
    base_output_path: Path | str = "outputs/eval/"
    

class SpineEval(RadianceFieldEval):
    def __init__(
        self,
        cfg: SpineEvalConfig = SpineEvalConfig(),
    ):
        # init super-class
        super().__init__(cfg=cfg)
        
        # semantic encoders
        self.sem_encoders: dict = {}
        
        if cfg.enable_semantic_encoders:
            # load semantic encoders
            self._load_semantic_encoders()
        
        # initialize segmentation metrics
        self._init_metrics()
        
    def _load_semantic_encoders(self):
        """
        Initializes the semantic encoders
        """   
        for enc_name, enc_config in tqdm(self.config.sem_encoders.items(),
                                         desc="Initializing the semantic encoders"):
            # initializes the semantic encoders
            self.sem_encoders[enc_name] = instantiate(enc_config, _recursive_=True)
    
    def _init_metrics(
        self,
    ):
        """
        Initialize metrics, e.g., SSIM, LPIPS, PSNR
        """
            
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
        # segmentation metrics
        self.supported_segmentation_metrics = {
            "miou",
            "accuracy",
            "psnr",
            "lpips",
            "ssim",
            "binary_cross_entropy",
        }
        
    # ---------------------------------- #
    # Evaluate Semantics PCA             #
    # ---------------------------------- #

    def evaluate_semantics_pca(
        self,
        encoder: BaseImageEncoder = None,
        rad_field: dict[str, RFModel] = None,
        images: torch.Tensor = None,
        cameras: torch.Tensor = None,
        visualize_results: bool = None,
        save_results: bool = None,
        process_geometry: bool = None,
        gradient_sobel_norm_threshold: float = None,
        num_pca_components: int = None,
        base_output_path: Path | str = None,
    ):
        # set options
        if rad_field is None:
            rad_field = self.rad_field
            
        if base_output_path is None:
            base_output_path = Path(f"{self.config.base_output_path}/pca_sobel_edge")
            
        if save_results is None:
            save_results = self.config.save_results
            
        # make directory, if necessary
        base_output_path = Path(base_output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        # output
        output: dict = {}
        
        # print
        self.console.rule(":rocket: Evaluating semantics...", style="#6d3cff")

        for scene, rf_fd_dict in rad_field.items():
            # create dict
            output[scene] = {}
                
            for rf_name, rf_fd in rf_fd_dict.items():

                if "base" in rf_name:
                    # base radiance field
                    continue
                
                # update the base output path
                bs_output_path = Path(f"{base_output_path}/{scene}/pca")
                
                # get results
                scene_res = self._evaluate_rf_semantics_pca(
                    encoder=encoder,
                    rad_field=rf_fd,
                    images=images,
                    cameras=cameras[scene],
                    visualize_results=visualize_results,
                    save_results=save_results,
                    process_geometry=process_geometry,
                    gradient_sobel_norm_threshold=gradient_sobel_norm_threshold,
                    num_pca_components=num_pca_components,
                    base_output_path=bs_output_path,
                )
                
                # collate the geometry stats            
                # RGB
                rgb_geom_stats = [
                        val["num_edges"]
                        for stats in scene_res["geometry"]
                        for key, val in stats.items()
                        if key.lower() in "rgb"
                    ]

                # semantic
                sem_geom_stats = [
                    val["num_edges"]
                    for stats in scene_res["geometry"]
                    for key, val in stats.items()
                    if key.lower() in "sem"
                ]
                
                # compute the summary statistics (mean and standard deviation)
                scene_res["summary_stats"] = {
                    "timing": {
                        "mean": np.nanmean(scene_res["timing"]),
                        "std": np.nanstd(scene_res["timing"]),
                    },
                    "sobel_edge": {
                        "rgb": {
                            "mean": np.nanmean(rgb_geom_stats, axis=0).tolist(),
                            "std": np.nanstd(rgb_geom_stats, axis=0).tolist(),
                        },
                        "sem": {
                            "mean": np.nanmean(sem_geom_stats, axis=0).tolist(),
                            "std": np.nanstd(sem_geom_stats, axis=0).tolist(),
                        },
                    }
                }
                    
                # insert the results
                output[scene][rf_name] = scene_res

                # also save for each scene
                if save_results:
                    with open(f"{base_output_path}/stats_{scene}_{rf_name}.json", "w") as fh:
                        json.dump(scene_res, fh, indent=4)
            
        # print
        self.console.rule("", style="#6d3cff")
        
        # save results
        if save_results:
            with open(f"{base_output_path}/stats.json", "w") as fh:
                json.dump(output, fh, indent=4)
                
        return output
            
    def _evaluate_rf_semantics_pca(
        self,
        encoder: BaseImageEncoder = None,
        rad_field: RFModel = None,
        images: torch.Tensor = None,
        cameras: torch.Tensor = None,
        visualize_results: bool = None,
        save_results: bool = None,
        process_geometry: bool = None,
        gradient_sobel_norm_threshold: float = None,
        num_pca_components: int = None,
        base_output_path: Path | str = None,
    ):
        """
        Runs Principal Component Analysis (PCA) for the semantic fields
        """
        
        # input-checking
        assert images is not None or cameras is not None, "PCA requires an image or camera pose!"

        # set the options
        # set encoder 
        if encoder is None and rad_field is not None:
            encoder = rad_field.pipeline.datamanager.img_semantics_interpolator.model
            
        if visualize_results is None:
            visualize_results = self.config.visualize_results
            
        if process_geometry is None:
            process_geometry = self.config.process_pca_geometry
            
        if gradient_sobel_norm_threshold is None:
            gradient_sobel_norm_threshold = self.config.gradient_sobel_norm_threshold
            
        if num_pca_components is None:
            num_pca_components = self.config.num_pca_components
            
        # make directory, if necessary
        base_output_path = Path(base_output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        # helper function to get PCA
        def get_pca(self, img, img_sem, img_idx):
            # save results
            if save_results:
                # output path
                output_path = Path(f"{base_output_path}/pca_{img_idx}.png")
        
            # compute the PCA
            elapsed_time, geometry_stats = self._compute_semantics_pca(
                encoder=encoder,
                cam_rgb=img,
                rend_sem_features=img_sem,
                visualize_results=visualize_results,
                save_results=save_results,
                process_geometry=process_geometry,
                gradient_sobel_norm_threshold=gradient_sobel_norm_threshold,
                num_pca_components=num_pca_components,
                output_path=output_path,
            )
            
            # append timing
            return elapsed_time, geometry_stats
        
        # print
        self.console.log(f":green_book: Saving images to {base_output_path}")
    
        # timing and geoemtry stats
        timing_stats: list = []
        geometry_stats: list = []
            
        if cameras is not None:
            # move to device
            cameras = cameras.to(self.device)
        
            # render images and semantics
            for cam_idx, cam in enumerate(tqdm(cameras, desc="Rendering images/semantics")):
                # render the RGB observation
                cam_outputs = rad_field.render(
                    cameras=cameras[cam_idx : cam_idx + 1], 
                    pose=None, 
                    compute_semantics=True,
                )
                
                # append images
                img = cam_outputs["rgb"]
                img_sem = cam_outputs["img_semantics"]
                
                # compute the PCA
                elapsed_time, geom_stats = get_pca(
                    self,
                    img=img,
                    img_sem=img_sem,
                    img_idx=cam_idx,
                )
                
                # append
                timing_stats.append(elapsed_time)
                geometry_stats.append(geom_stats)
        else:
            # compute the PCA
            for img_idx, img in enumerate(tqdm(images, desc="Evaluating PCA")):
                # compute the PCA
                elapsed_time, geom_stats = get_pca(
                    self,
                    img=img,
                    img_sem=None,
                    img_idx=img_idx,
                )
                
                # append
                timing_stats.append(elapsed_time)
                geometry_stats.append(geom_stats)
                
        # output
        output = {
            "encoder": encoder.name,
            "timing": timing_stats,
            "geometry": geometry_stats,
        }
            
        return output
    
    @torch.no_grad()
    def _compute_semantics_pca(
        self,
        encoder: BaseImageEncoder,
        cam_rgb: torch.Tensor,
        rend_sem_features: torch.Tensor = None,  # rendered semantic features
        visualize_results: bool = False,
        save_results: bool = False,
        process_geometry: bool = False,
        gradient_sobel_norm_threshold: list[float] = [1e-2],
        num_pca_components: int = 4,
        output_path: Path | str = "outputs/semantics/pca",
    ):
        """
        Compute a coarse estimate of the camera's pose from an RGB image.
        
        Arguments:
        cam_rgb: torch.Tensor, RGB image with shape H x W x C
        ...
        """
            
        # make directory, if necessary
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # start time
        start_time = time.perf_counter()
        
        if rend_sem_features is None:
            if "vggt" in encoder.name:
                 # compute the semantic features of the image
                cam_rgb_sem_features = encoder.encode_image(
                    [cam_rgb.moveaxis(-1, -3)],  # [C x H x W, ]
                )
                
                # for intermediate-layer features
                cam_rgb_sem_features = cam_rgb_sem_features[1].squeeze().half()
                
            elif "dino" in encoder.name:
                # output semantic features size (H x W) for the DINOv2 encoder
                out_img_size = ((torch.floor(torch.tensor(cam_rgb.shape[:2]) / 14) - 15) * 14).int().cpu().numpy().tolist()

                # compute the semantic features of the image
                cam_rgb_sem_features = encoder.encode_image(
                    cam_rgb.moveaxis(-1, -3).unsqueeze(0),  # B x C x H x W
                    img_size=out_img_size,
                ).squeeze().half()
                
            # interpolate semantic featires to match the dimension of the RGB image
            torch_img_resize = tv_transforms.Resize(cam_rgb.shape[:2])
            cam_rgb_sem_features = torch_img_resize(cam_rgb_sem_features.moveaxis(-1, -3)).moveaxis(-3, -1)
        else:
            cam_rgb_sem_features = rend_sem_features
            
        # flattened semantic features
        cam_rgb_sem_features_rs = cam_rgb_sem_features.reshape(-1, cam_rgb_sem_features.shape[-1]).float()
        
        # PCA
        U_pca_lr, S_pca_lr, V_pca_lr = torch.pca_lowrank(
            A=cam_rgb_sem_features_rs,
            q=num_pca_components,
            center=True, # default
            niter=20, # default
        )
        
        # project to the first-3 principal directions
        semantic_img_proj = cam_rgb_sem_features_rs @ V_pca_lr[:, :num_pca_components]
        
        # extract background and foreground
        semantic_img_proj_bgd = semantic_img_proj[..., 0:1].detach().clone()
        semantic_img_proj = semantic_img_proj[..., 0:3]
        
        if self.config.enable_outlier_rejection_in_pca:
            # outlier rejection using median (taken from Nerfstudio)
            d = torch.abs(semantic_img_proj - torch.median(semantic_img_proj, dim=0).values)
            mdev = torch.median(d, dim=0).values
            s = d / mdev
            m = 2.0  # this is a hyperparam controlling how many std dev outside for outliers
            rins =semantic_img_proj[s[:, 0] < m, 0]
            gins = semantic_img_proj[s[:, 1] < m, 1]
            bins = semantic_img_proj[s[:, 2] < m, 2]
            
            # scale values
            semantic_img_proj[:, 0] -= rins.min()
            semantic_img_proj[:, 1] -= gins.min()
            semantic_img_proj[:, 2] -= bins.min()
            
            semantic_img_proj[:, 0] /= (rins.max() - rins.min())
            semantic_img_proj[:, 1] /= (gins.max() - gins.min())
            semantic_img_proj[:, 2] /= (bins.max() - bins.min())
            
            # map to the image space in [0, 1]
            semantic_img_proj = torch.clamp(semantic_img_proj, 0, 1)
        else:
            # map to the image space in [0, 1]
            semantic_img_proj -= semantic_img_proj.min(dim=0, keepdim=True)[0]
            semantic_img_proj /= semantic_img_proj.max(dim=0, keepdim=True)[0]
        
        # extract the source and target images
        semantic_img_proj = (
            semantic_img_proj.reshape(*cam_rgb_sem_features.shape[:-1], 3) # num_pca_components - 1)
        )
        
        # background
        semantic_img_proj_bgd = (
            semantic_img_proj_bgd.reshape(*cam_rgb_sem_features.shape[:-1], 1)
        )
        
        # elapsed time
        elapsed_time = time.perf_counter() - start_time
        
        # visualize the RGB and semantic images
        # RGB image
        rgb_img = Image.fromarray((cam_rgb.cpu().numpy() * 255).astype(np.uint8))
        # semantic image
        sem_img = Image.fromarray((semantic_img_proj.cpu().numpy() * 255).astype(np.uint8))
        
        if visualize_results:
            # show images
            rgb_img.show()
            sem_img.show()
            
        # encoder name
        enc_name = 'encoder' if encoder is None else f"{encoder.name}_{encoder.config.feature_key}"
        
        # save the images
        if save_results:
            base_path, ext = os.path.splitext(output_path)
            rgb_img.save(f"{base_path}_rgb{ext}")
            sem_img.save(f"{base_path}_{enc_name}_sem{ext}")
        
        # examine geomery (edge extraction), if requested
        if process_geometry:
            # images to process
            img_inp = {
                "rgb": rgb_img,
                "sem": sem_img,
            }
        
            # process geometry
            geometry_stats = self._process_geometry(
                img=img_inp,
                encoder_name=enc_name,
                gradient_sobel_norm_threshold=gradient_sobel_norm_threshold,
                visualize_results=visualize_results,
                save_results=save_results,
                output_path=output_path,
            )
        else:
            geometry_stats = {}
        
        del semantic_img_proj
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return elapsed_time, geometry_stats
    
    def _process_geometry(
        self,
        img,
        gradient_sobel_norm_threshold: list[float] = [1e-2],
        encoder_name: str = "encoder",
        visualize_results: bool = False,
        save_results: bool = False,
        output_path: Path | str = "outputs/semantics/sobel",
    ):
        """
        Extract edges from images and return stats
        """
        # TODO: type-checking for inputs
        
        # output path
        output_path = Path(output_path)
        
        # geometry stats
        geometry_stats = {}
        
        for img_key, img in img.items():
            # map to grayscale
            img = img.convert("L")
            
            # cast to tensor
            img = torch.tensor(np.asarray(img)[..., None] / 255.0).float().to(self.device)
            img = img.moveaxis(-1, 0)[None]
           
            # image gradient
            img_sobel_grad = sobel_gradients(img)

            # compute the norm of the gradient
            img_sobel_gnorm = torch.linalg.norm(img_sobel_grad, dim=1, keepdim=True)

            # initialize stats
            img_stats = {
                "thresholds": gradient_sobel_norm_threshold,
                "num_edges": [],
                "norm_grad": torch.linalg.norm(img_sobel_gnorm).item(),
                "post_threshold_norm_grad": [],
            }

            for threshold in gradient_sobel_norm_threshold:
                # visualize the gradient
                sobel_img = torch.zeros_like(img_sobel_grad[:, 0:1, ...])
                # sobel_img[img_sobel_gnorm > gradient_sobel_norm_threshold] = 1
                sobel_img[img_sobel_gnorm > threshold] = 1
                # convert to PIL Image (grayscale)
                sobel_img = Image.fromarray((sobel_img.squeeze(dim=0).moveaxis(0, -1) * 255).detach().cpu().numpy().astype(np.uint8).squeeze())

                # compute the stats
                img_stats["num_edges"].append((img_sobel_gnorm > threshold).sum().item())
                img_stats["post_threshold_norm_grad"].append(torch.linalg.norm(img_sobel_gnorm[img_sobel_gnorm > threshold]).item())
  
                if visualize_results:
                    # show images
                    self.console.print(f"Sobel image (edges) for {img_key} for threshold {threshold}")
                    sobel_img.show()
            
                # save the images
                if save_results:
                    # base path
                    base_path = output_path.parent.parent
                    fname = output_path.stem
                    _, ext = os.path.splitext(output_path)
                    
                    # suffix for the filename
                    if "sem" in img_key.lower():
                        fname_suffix = f"{encoder_name}_sem"
                    else:
                        fname_suffix = img_key
                    
                    # file path
                    out_path = Path(f"{base_path}/sobel_edge/{fname}_{fname_suffix}_{threshold}{ext}")
                    
                    # make directory. if necessary
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # save image
                    sobel_img.save(out_path)

            # cache the stats
            geometry_stats[img_key] = img_stats
                
        return geometry_stats
    
    # ---------------------------------- #
    # Evaluate Semantic Segmentation     #
    # ---------------------------------- #
    
    def evaluate_semantic_segmentation(
        self,
        encoder: BaseImageEncoder = None,
        rad_field: dict[str, RFModel] = None,
        images: torch.Tensor = None,
        cameras: torch.Tensor = None,
        object_queries: dict[str, dict] = None,
        negative_queries: str = None,
        filter_by_max_score: bool = None,
        visualize_results: bool = None,
        save_results: bool = None,
        pred_mask_threshold: float = None,
        pred_mask_quantile_q: float = None,
        base_output_path: Path | str = None,
    ):
        """
        Evaluates semantic segmentation for the semantic fields
        """
        # set options
        if rad_field is None:
            rad_field = self.rad_field
            
        if base_output_path is None:
            base_output_path = Path(f"{self.config.base_output_path}/segmentation")
            
        if save_results is None:
            save_results = self.config.save_results
            
        # make directory, if necessary
        base_output_path = Path(base_output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        # output
        output: dict = {}
        
        # print
        self.console.rule(":rocket: Evaluating semantic segmentation...", style="#ff75d8")


        for scene, rf_fd_dict in rad_field.items():
            # create dict
            output[scene] = {}
                
            for rf_name, rf_fd in rf_fd_dict.items():
                if "base" in rf_name:
                    # base radiance field
                    continue
                
                # update the base output path
                bs_output_path = Path(f"{base_output_path}/{scene}/semantic_segmentation")
                
                # get results
                scene_res = self._evaluate_rf_semantic_segmentation(
                    encoder=encoder,
                    rad_field=rf_fd,
                    images=images,
                    cameras=cameras[scene],
                    object_queries=object_queries[scene],
                    negative_queries=negative_queries,
                    filter_by_max_score=filter_by_max_score,
                    visualize_results=visualize_results,
                    save_results=save_results,
                    pred_mask_threshold=pred_mask_threshold,
                    pred_mask_quantile_q=pred_mask_quantile_q,
                    base_output_path=bs_output_path,
                )
                
                # insert the results
                output[scene][rf_name] = scene_res

                # also save for each scene
                if save_results:
                    with open(f"{base_output_path}/stats_{scene}_{rf_name}.json", "w") as fh:
                        json.dump(scene_res, fh, indent=4)
            
        # print
        self.console.rule("", style="#ff75d8")
        
        # save results
        if save_results:
            with open(f"{base_output_path}/stats.json", "w") as fh:
                json.dump(output, fh, indent=4)
                
        return output
            
    def _evaluate_rf_semantic_segmentation(
        self,
        encoder: BaseImageEncoder = None,
        rad_field: RFModel = None,
        images: torch.Tensor = None,
        cameras: torch.Tensor = None,
        object_queries: list[str] = None,
        negative_queries: str = None,
        filter_by_max_score: bool = None,
        visualize_results: bool = None,
        save_results: bool = None,
        pred_mask_threshold: float = None,
        pred_mask_quantile_q: float = None,
        base_output_path: Path | str = None,
    ):
        """
        Evaluates semantic segmentation in a semantic radiance field
        """
        # TODO: additional input-checking
        # input-checking
        assert images is not None or cameras is not None, "Semantic segmentation requires an image or camera pose!"
        
        # set the options
        # set encoder 
        if encoder is None and rad_field is not None:
            encoder = rad_field.pipeline.datamanager.img_semantics_interpolator.model
            
        if visualize_results is None:
            visualize_results = self.config.visualize_results
            
        if negative_queries is None:
            negative_queries = self.config.negative_queries
            
        if pred_mask_threshold is None:
            pred_mask_threshold = self.config.pred_mask_threshold
            
        if pred_mask_quantile_q is None:
            pred_mask_quantile_q = self.config.pred_mask_quantile_q
            
        if filter_by_max_score is None:
            filter_by_max_score = self.config.filter_by_max_score
            
        # make directory, if necessary
        base_output_path = Path(base_output_path)
        base_output_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: Refactor to support definition from function arguments
        # initialize the object detection and segmentation models to compute the ground-truth masks
        gdino, sam2_predictor, sam2_mask_generator = self._init_segmentation_models()

        # print
        self.console.log(f":green_book: Saving images to {base_output_path}")
    
        # errors
        segmentation_error = {}

        # segmentation metrics
        for met in self.supported_segmentation_metrics:
            segmentation_error[met] = {}
            
        # indicator for one metric
        samp_metric_name = list(self.supported_segmentation_metrics)[0]
            
        if cameras is not None:
            # move to device
            cameras = cameras.to(self.device)
        
            # render images and semantics
            for cam_idx, cam in enumerate(tqdm(cameras, desc="Running semantic segmentation")):
                # save results
                if save_results:
                    # output path
                    output_path = Path(f"{base_output_path}/img_{cam_idx}.png")
        
                for obj_idx, obj_name in enumerate(object_queries):
                    # get results
                    seg_metrics = self._eval_object_semantic_segmentation(
                        encoder=encoder,
                        rad_field=rad_field,
                        images=None,
                        camera=cameras[cam_idx : cam_idx + 1],
                        object_query=obj_name,
                        negative_query=negative_queries,
                        gdino_model=gdino,
                        sam2_predictor=sam2_predictor,
                        filter_by_max_score=filter_by_max_score,
                        visualize_results=visualize_results,
                        save_results=save_results,
                        pred_mask_threshold=pred_mask_threshold,
                        pred_mask_quantile_q=pred_mask_quantile_q,
                        output_path=output_path,
                    )
                    
                    if seg_metrics is not None:
                        if not obj_name in segmentation_error[samp_metric_name]:
                            for met in self.supported_segmentation_metrics:
                                segmentation_error[met][obj_name] = []

                        # cache results
                        for key, val in seg_metrics.items():
                            segmentation_error[key][obj_name].append(val)
        else:
            raise NotImplementedError("Support for image inputs not yet implemented!")
                
        # print the results
        seg_stats, composite_seg_stats = self._display_segmentation_results(
            segmentation_error=segmentation_error,
            object_queries=object_queries,
            caption=f"{base_output_path}, Encoder: {encoder.name}"
        )
        
        # cast to list
        for key_up in segmentation_error.keys():
            for key_down in segmentation_error[key_up]:
                segmentation_error[key_up][key_down] = np.array(
                    segmentation_error[key_up][key_down]
                ).tolist()
        
        # output
        output = {
            "encoder": encoder.name,
            "summary_stats": composite_seg_stats,
            "obj_summary_stats": seg_stats,
            "all_stats": segmentation_error,
        }
            
        return output
    
    def _eval_object_semantic_segmentation(
        self,
        encoder: BaseImageEncoder = None,
        rad_field: RFModel = None,
        images: torch.Tensor = None,
        camera: torch.Tensor = None,
        object_query: str = None,
        negative_query: str = None,
        gdino_model: torch.nn.Module = None,
        sam2_predictor: torch.nn.Module = None,
        filter_by_max_score: bool = None,
        visualize_results: bool = None,
        save_results: bool = None,
        pred_mask_threshold: float = None,
        pred_mask_quantile_q: float = None,
        output_path: Path | str = "outputs/segmentation/img.png",
    ):
        """
        Evaluate semantic segmenation for a query object
        """
        # set the semantic query
        # positive query
        positive_texts = object_query
        rad_field.pipeline.model.viewer_utils.handle_language_queries(
            raw_text=positive_texts, is_positive=True
        )

        # negative query
        rad_field.pipeline.model.viewer_utils.handle_language_queries(
            raw_text=negative_query, is_positive=False
        )

        # render a semantic similarity image
        rf_outputs = rad_field.render(
            pose=camera.camera_to_worlds.squeeze(),
            compute_semantics=True,
        )

        if save_results:
            # output path
            base_path, ext = os.path.splitext(output_path)
            
            if "." not in ext:
                ext = ".png"
            
            # rendered images
            rendered_rgb = rf_outputs["rgb"].cpu()[None]
            # rendered_seg = rf_outputs["lang_similarity_GUI"].cpu()[None]
            rendered_seg = apply_colormap(rf_outputs["lang_similarity"], colormap="turbo")
            rendered_seg = rendered_seg.detach().cpu()[None]
            
            # rendered images
            s_img = {
                "rgb": rendered_rgb,
                "pred_sim": rendered_seg,
            }

            # save the images
            for s_key, s_img in s_img.items():
                # image output path
                img_output_path: str = f"{base_path}_{s_key}_{encoder.name}"
                
                if "sim" in s_key:
                    img_output_path = f"{img_output_path}_{object_query.replace(' ', '_')}"
                
                # save the image
                Image.fromarray(
                    (s_img.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                ).save(Path(f"{img_output_path}{ext}"))

        # get the ground-truth mask and masked image
        gt_mask, gt_mask_img, post_results = self._get_ground_truth_semantic_segmentation(
            rgb_image=rf_outputs["rgb"],
            object_query=object_query,
            gdino_model=gdino_model,
            sam2_predictor=sam2_predictor,
            filter_by_max_score=filter_by_max_score,
            visualize_results=visualize_results,
            save_results=save_results,
        )
        
        # early termination
        if gt_mask is None:
            return None
        
        # ground-truth mask probabilies in {0, 1}
        gt_mask_prob = torch.tensor(gt_mask).float().to(self.device)
        
        if save_results:
            # save similarity of ground-truth mask
            gt_mask_rgb = apply_colormap(
                gt_mask_prob,
                colormap="turbo",
            )
            
            # image
            s_img = {
                "gt_sim": gt_mask_rgb,
            }

            # save the images
            for s_key, s_img in s_img.items():
                # image output path
                img_output_path: str = f"{base_path}_{s_key}_{encoder.name}"
                
                if "sim" in s_key:
                    img_output_path = f"{img_output_path}_{object_query.replace(' ', '_')}"
                
                # save the image
                Image.fromarray(
                    (s_img.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                ).save(Path(f"{img_output_path}{ext}"))
        
        # compute the quantile for the predicted mask
        pred_mask_prob = rf_outputs["lang_similarity"].to(self.device)
        pred_mask_quantile = torch.quantile(
            pred_mask_prob, pred_mask_quantile_q
        )

        # predicted segmentation mask
        pred_mask = (
            # (outputs["lang_similarity"] == outputs["lang_similarity"].max())
            torch.logical_and(
                pred_mask_prob > pred_mask_threshold,
                pred_mask_prob > pred_mask_quantile,
            )
            .cpu().numpy()
        )
        
        if visualize_results or save_results:
            self.console.log("Predicted Mask")
            pred_mask_img = show_masks(
                Image.fromarray(
                    (rf_outputs["rgb"].squeeze().detach().cpu().numpy() * 255
                ).astype(np.uint8)),
                pred_mask.squeeze()[None],
                np.ones(gt_mask.shape[0],),
                box_coords=post_results["bboxes"],
                input_labels=post_results["labels"],
            )
        
        # close all images
        plt.close("all")
            
        if save_results:
            # rendered images
            s_img = {
                "gt_mask": gt_mask_img,
                "pred_mask": pred_mask_img,
            }

            # save the images
            for s_key, s_img in s_img.items():
                # image output path
                img_output_path: str = f"{base_path}_{s_key}_{encoder.name}"
                
                # insert the object name
                img_output_path = f"{img_output_path}_{object_query.replace(' ', '_')}"
                
                # save the image
                s_img.savefig(Path(f"{img_output_path}{ext}"), bbox_inches='tight')
                
        # # #
        # # # Compute the object segmentation metrics
        # # #
        
        seg_metrics = self._compute_object_semantic_segmentation_metrics(
            pred_mask_prob=pred_mask_prob,
            gt_mask_prob=gt_mask_prob,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
        )

        # # compute the mIoU
        # img_miou = compute_mIoU(
        #     predicted_mask=pred_mask.round().astype(bool),
        #     ground_truth_mask=gt_mask.round().astype(bool),
        # )

        # # compute the accuracy
        # img_accuracy = compute_segmentation_accuracy(
        #     predicted_mask=pred_mask.round().astype(bool),
        #     ground_truth_mask=gt_mask.round().astype(bool),
        # )

        return seg_metrics
    
    def _compute_object_semantic_segmentation_metrics(
        self,
        pred_mask_prob: np.ndarray | torch.Tensor,
        gt_mask_prob: np.ndarray | torch.Tensor,
        pred_mask: np.ndarray | torch.Tensor,
        gt_mask: np.ndarray | torch.Tensor,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Compute the metrics for object semantic segmentation outputs
        """
        # type-checking
        if isinstance(pred_mask_prob, np.ndarray):
            pred_mask_prob = torch.tensor(pred_mask_prob).float().to(device)
            
        if isinstance(gt_mask_prob, np.ndarray):
            gt_mask_prob = torch.tensor(gt_mask_prob).float().to(device)
            
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
            
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy()
           
        # compute the metrics
        
        # compute the mIoU
        img_miou = compute_mIoU(
            predicted_mask=pred_mask.round().astype(bool),
            ground_truth_mask=gt_mask.round().astype(bool),
        )

        # compute the accuracy
        img_accuracy = compute_segmentation_accuracy(
            predicted_mask=pred_mask.round().astype(bool),
            ground_truth_mask=gt_mask.round().astype(bool),
        )
        
        # compute the binary cross-entropy loss
        bce = torch.nn.functional.binary_cross_entropy(
            input=pred_mask_prob,
            target=gt_mask_prob,
        ).detach().cpu().numpy()
        
        # convert probabilities to images
        gt_mask_rgb = apply_colormap(
            gt_mask_prob, colormap="turbo",
        ).to(device)
        pred_mask_rgb = apply_colormap(
            pred_mask_prob, colormap="turbo",
        ).to(device)
        
        # reshape from [H, W, C] to [1, C, H, W] for SSIM, PSNR, LPIPS
        gt_mask_rgb = torch.moveaxis(gt_mask_rgb, -1, 0)[None, ...]
        pred_mask_rgb = torch.moveaxis(pred_mask_rgb, -1, 0)[None, ...]

        # compute the SSIM, PSNR, LPIPS
        ssim = self.ssim(gt_mask_rgb, pred_mask_rgb).detach().cpu().numpy()
        psnr = self.psnr(gt_mask_rgb, pred_mask_rgb).detach().cpu().numpy()
        lpips = self.lpips(gt_mask_rgb, pred_mask_rgb).detach().cpu().numpy()
        
        # output
        output = {
            "miou": img_miou,
            "accuracy": img_accuracy,
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips,
            "binary_cross_entropy": bce,
        }
        
        return output
        
    def _init_segmentation_models(self):
        """
        Initializes the object detection and segmentation models
        """
        # load GDINO
        gdino = load_gdino()
        
        # load SAM-2
        sam2_predictor, sam2_mask_generator = load_sam2()
        
        return gdino, sam2_predictor, sam2_mask_generator
    
    def _get_ground_truth_semantic_segmentation(
        self,
        rgb_image: torch.Tensor = None,
        object_query: str = None,
        gdino_model: torch.nn.Module = None,
        sam2_predictor: torch.nn.Module = None,
        filter_by_max_score: bool = None,
        visualize_results: bool = None,
        save_results: bool = None,
    ):
        """
        Evaluate semantic segmenation for a query object
        """

        # #
        # #  GroundingDINO
        # #

        # image for GroundingDINO
        image_source_dino, image_tensor_dino = (
            rgb_image.cpu().numpy() * 255
        ).astype(np.uint8), rgb_image.permute(-1, 0, 1)

        # RGB image as PIL Image
        image_rgb_pil = Image.fromarray(image_source_dino)

        # TODO: add support for runtime specification of thresholds (box_threshold, text_threshold, caption)
        BOX_TRESHOLD = self.config.gdino_config["BOX_TRESHOLD"]
        TEXT_TRESHOLD = self.config.gdino_config["TEXT_TRESHOLD"]
        TEXT_PROMPT = object_query
        
        # GroundingDINO prediction
        boxes, logits, phrases = predict(
            model=gdino_model,
            image=image_tensor_dino,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=self.device,
        )

        if self.config.verbose_print:
            print(f"GroundingDINO output logits: {logits} and phrases {phrases}")

        # post-process the outputs
        gdino_val_idx = [
            idx
            for idx, det_label in enumerate(phrases)
            if TEXT_PROMPT in det_label
        ]

        if len(phrases) == 0 or len(gdino_val_idx) == 0:
            if self.config.verbose_print:
                self.console.log(
                    f"{object_query.capitalize()} was not detected in the image!"
                )

        # post-process the outputs
        boxes = boxes[gdino_val_idx]
        logits = logits[gdino_val_idx]
        phrases = [phrases[g_id] for g_id in gdino_val_idx]
        
        if len(logits) == 0:
            return None, None, None
        
        if filter_by_max_score:
            # take the maximum score
            arg_max = np.argmax(logits)
            boxes = boxes[arg_max : arg_max + 1]
            logits = logits[arg_max : arg_max + 1]
            phrases = phrases[arg_max : arg_max + 1]
            
        if self.config.verbose_print:
            print(f"GroundingDINO processed output logits: {logits}, phrases {phrases}, boxes {boxes}")

        # visualize the results
        if visualize_results:
            annotated_frame = annotate(
                image_source=image_source_dino,
                boxes=boxes,
                logits=logits,
                phrases=phrases,
            )
            annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

            # show image
            Image.fromarray(annotated_frame).show()

        # # #
        # # # Post-process the Bounding-Boxes
        # # #

        # image height and width
        H, W = image_source_dino.shape[:2]

        # convert the bounding-boxes to xyxy format
        if boxes.max() < 1:
            boxes_dino_cxcywh = boxes * torch.Tensor([W, H, W, H])
            boxes_dino_xyxy = box_convert(
                boxes=boxes_dino_cxcywh, in_fmt="cxcywh", out_fmt="xyxy"
            ).numpy()
        else:
            boxes_dino_xyxy = boxes.numpy()

        # postprocess the bounding box
        post_bbox = post_process_bbox(
            bboxes=boxes_dino_xyxy.tolist(),
            labels=phrases,
            img_dim=[H, W],
            max_box_dim_threshold=0.9,
        )

        # visualize the results
        post_results = {
            "bboxes": np.array(list(post_bbox.keys())),
            "labels": [val["caption"].lower() for val in post_bbox.values()],
            "cap_len": [val["caption_len"] for val in post_bbox.values()],
        }

        if visualize_results:
            plot_bbox(image_rgb_pil, post_results)

        # #
        # #  SAM2
        # #
        
        # Mask generation per Bounding Box
        image_arr = image_source_dino

        # set the image
        sam2_predictor.set_image(image_arr)

        # get mask
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_dino_xyxy,
            multimask_output=False,
        )

        # masks and scores
        if len(masks.shape) > 3:
            masks, scores = masks.squeeze(), scores.squeeze()

        # visualize result
        if visualize_results:
            self.console.log("Ground-Truth Mask")
            show_masks(
                image_rgb_pil,
                masks,
                scores,
                box_coords=post_results["bboxes"],
                input_labels=post_results["labels"],
            )

        # ground-truth mask
        gt_mask = masks.sum(axis=0, keepdims=True)
        gt_mask[gt_mask >= 1.0] = 1.0

        if visualize_results or save_results:
            self.console.log("Aggregated Ground-Truth Mask")
            gt_mask_img = show_masks(
                image_rgb_pil,
                gt_mask,
                scores,
                box_coords=post_results["bboxes"],
                input_labels=post_results["labels"],
            )
        
        # close all images
        plt.close("all")

        # ground-truth mask
        gt_mask = gt_mask.squeeze()[..., None]
        
        return gt_mask, gt_mask_img, post_results 

    def _display_segmentation_results(
        self,
        segmentation_error: dict[str, dict[str, list]],
        object_queries: list[str],
        caption: str,
    ):
        """
        Display summary statistics for semantic segmentation results
        """
        # cast to array
        for key_up in segmentation_error.keys():
            for key_down in segmentation_error[key_up]:
                segmentation_error[key_up][key_down] = np.array(
                    segmentation_error[key_up][key_down]
                )

        # compute the composite statistics
        composite_seg_error = {}
        for key_up in segmentation_error.keys():
            if key_up not in composite_seg_error:
                composite_seg_error[key_up] = []

            for key_down in segmentation_error[key_up]:
                composite_seg_error[key_up].extend(segmentation_error[key_up][key_down])

        ## %%
        # print the results
        segmentation_stats = copy.deepcopy(segmentation_error)

        for idx, key_up in enumerate(segmentation_error.keys()):
            for key_down in segmentation_error[key_up].keys():
                # summary statistics
                segmentation_stats[key_up][
                    key_down
                ] = f"""
                    {np.nanmean(segmentation_error[key_up][
                    key_down
                ]):.4e} \u00B1 {np.nanstd(segmentation_error[key_up][
                    key_down
                ]):.4e}
                """

        # display summary statistics
        summ_table = pd.DataFrame(segmentation_stats.values())

        # rename index
        summ_table.rename(
            index=dict(
                zip(
                    range(0, len(segmentation_stats.keys())),
                    [key.capitalize() for key in list(segmentation_stats.keys())],
                )
            ),
            inplace=True,
        )

        # display stats in a table
        num_runs_caption = "Number of Runs.\n"

        for obj_name in object_queries:
            if obj_name in segmentation_error["miou"]:
                num_runs_caption += (
                    f"{obj_name}: {segmentation_error['miou'][obj_name].size}\n"
                )

        self.console.log(num_runs_caption)

        try:
            display(
                summ_table.style.set_caption(
                    caption,
                ).set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [("font-weight", "bold"), ("font-size", "14px")],
                        }
                    ]
                )
            )
        except NameError:
            print(f"An IPython Environment was not detected!")

        # print the composite statistics
        self.console.rule(f"Composite Statistics")

        # composite statistics
        composite_segmentation_stats = {}

        for idx, key_up in enumerate(composite_seg_error.keys()):
            # summary statistics
            composite_segmentation_stats[
                key_up
            ] = f"""
                {np.nanmean(composite_seg_error[key_up]):.4e} \u00B1 {np.nanstd(composite_seg_error[key_up]):.4e}
            """

        # display summary statistics
        summ_table = pd.DataFrame(composite_segmentation_stats.values())

        # rename index
        summ_table.rename(
            index=dict(
                zip(
                    range(0, len(composite_segmentation_stats.keys())),
                    [
                        key.capitalize()
                        for key in list(composite_segmentation_stats.keys())
                    ],
                )
            ),
            inplace=True,
        )

        try:
            display(
                summ_table.style.set_caption(
                    caption,
                ).set_table_styles(
                    [
                        {
                            "selector": "caption",
                            "props": [("font-weight", "bold"), ("font-size", "14px")],
                        }
                    ]
                )
            )
        except NameError:
            print(f"An IPython Environment was not detected!")

        # print the termination line
        self.console.rule("")
        
        return segmentation_stats, composite_segmentation_stats
    
    
    # ---------------------------------- #
    # Evaluate Radience Field Inversion  #
    # ---------------------------------- #

    def evaluate_rf_inversion(
            self,
            encoder: BaseImageEncoder = None,
            rad_field: dict[str, RFModel] = None,
            images: torch.Tensor = None, # TODO: load image
            cameras: torch.Tensor = None, # TODO: load camera poses
            save_results: bool = None,
            base_output_path: Path | str = None,
            compute_baseline_stats: bool = True,
        ):

        """
        Evaluates the radiance field inversion methods
        """
        
        # set options
        if rad_field is None:
            rad_field = self.rad_field
            
        if base_output_path is None:
            base_output_path = Path(f"{self.config.base_output_path}/rf_inversion")

        # make directory, if necessary
        base_output_path.mkdir(parents=True, exist_ok=True)
                
        # output
        output: dict = {}
        
        # print
        self.console.rule(":rocket: Evaluating rf inversion...", style="#3cf5ff")

        for scene, rf_fd_dict in rad_field.items():
            # create dict
            output[scene] = {}
            
            # update the base output path
            bs_output_path = Path(f"{base_output_path}/{scene}")
                
            # get base radiance field
            scene_base_rf = rf_fd_dict["base"]
            
            # completion flag for the baseline
            baseline_done: bool = False
                
            for rf_name, rf_fd in rf_fd_dict.items():
                if "base" in rf_name:
                    # base radiance field
                    continue
                
                # current setting for semantic rendering
                sem_render_setting = rf_fd.pipeline.model.config.override_compute_semantics_no_compute
                
                # disable semantic rendering
                rf_fd.pipeline.model.config.override_compute_semantics_no_compute = True
                
                # insert result
                output[scene][rf_name] = self._compute_rf_inversion(
                    encoder=encoder,
                    rad_field=rf_fd,
                    base_rad_field=scene_base_rf,
                    images=images,
                    cameras=cameras[scene],
                    save_results=save_results,
                    output_path=bs_output_path,
                )
                
                if compute_baseline_stats and not baseline_done:
                    # compute the baseline stats
                    output[scene]["baseline"] = self._run_rf_inversion_baselines(
                        rad_field=rf_fd,
                        base_rad_field=scene_base_rf,
                        images=images,
                        cameras=cameras[scene],
                        save_results=save_results,
                        output_path=bs_output_path,
                    )
                    baseline_done = True
                    
                # reset semantic rendering
                rf_fd.pipeline.model.config.override_compute_semantics_no_compute = sem_render_setting
                
        # print
        self.console.rule("", style="#3cf5ff")

        return output
    
    def _compute_rf_inversion(
        self,
        encoder: BaseImageEncoder,
        rad_field: RFModel,
        images: list[torch.Tensor],
        cameras: list[torch.Tensor],
        base_rad_field: RFModel = None,
        save_results: bool = False,
        output_path: Path | str = "outputs/eval/rf_inversion",
        print_stats: bool = True,
    ):
        # input type-checking
        # set encoder 
        if encoder is None and rad_field is not None:
            encoder = rad_field.pipeline.datamanager.img_semantics_interpolator.model

        # save results
        if save_results is None:
            save_results = self.config.save_results
            
        # update the output path
        output_path: Path = Path(f"{output_path}/{encoder.name}")
        
        # make directory, if necessary
        output_path.mkdir(parents=True, exist_ok=True)
                    
        # --- Step 1: Get GT camera poses and images ---
        if images is None:
            assert base_rad_field is not None, "A radiance field for image rendering/generation is required!"
            cameras, images = self._load_rf_inversion_data(
                base_rad_field=base_rad_field,
                cameras=cameras,
            )
        
        # --- Step 2: Encode images to get camera pose embeddings ---
        with torch.no_grad():
            # with torch.cuda.amp.autocast(dtype=spec_dtype):
            pred_cam_pose_embedding, _, _ = encoder.encode_image(
                images, 
                return_camera_feats=True,
                return_img_feats=False,
            )
            
        # --- Step 3: Run through decoder to get camera poses and errors ---
        # all errors
        pose_errors = []
        est_cam_poses = []
    
        # success flag (i.e., no error thrown)
        success_flags = []
        
        with torch.no_grad():
            # decode the camera pose
            cam_pose_dec_output = rad_field.pipeline.model.spine_field.decode_camera_pose(
                F.normalize(pred_cam_pose_embedding, p=2, dim=-1),
                use_expected_value=True,
                compute_uncertainty=True,
            )
            
        # unpack decoder output
        cam_pose_dec_trans = cam_pose_dec_output["pose_trans"]
        cam_pose_dec_rot = cam_pose_dec_output["pose_rot"]
        cam_pose_entropy = cam_pose_dec_output["entropy"].detach().cpu().numpy()

        for p_idx, pose in enumerate(tqdm(cam_pose_dec_trans, desc="computing error in pose estimates")):
            # map to an SE(3) pose
            est_cam_pose = torch.eye(4)
            # rotation
            est_cam_pose[:3, :3] = _lie_algebra_to_rotmat(cam_pose_dec_rot[p_idx])
            # translation
            est_cam_pose[:3, 3] = cam_pose_dec_trans[p_idx]
            est_cam_poses.append(est_cam_pose.cpu().numpy())
            
            # ground-truth pose
            gt_pose = cameras[p_idx].cpu().numpy()
            
            # pose error (rotation error, translation error)
            error = SE3error(gt_pose, est_cam_pose.cpu().numpy())[:2]
            
            # cache the error
            pose_errors.append(error)
            
            # success
            success = 1
            
            # cache the success
            success_flags.append(success)

            if print_stats:
                self.console.log(f"Idx: {p_idx}, SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}")
        

        # --- Step 4: Compute Pose Error Stats ---
        # mean and std error
        mean_pose_error = np.nanmean(pose_errors, axis=0)
        std_pose_error = np.nanstd(pose_errors, axis=0)
        
        # mean success rate (i.e., no error thrown)
        mean_success_rate = np.nanmean(success_flags, axis=0)
        
        # mean pose entropy
        mean_pose_entropy = np.nanmean(cam_pose_entropy, axis=0)
        std_pose_entropy = np.nanstd(cam_pose_entropy, axis=0)

        # --- Step 5: Save results ---
        if save_results:
            with open(f"{output_path}/pose_errors.json", "w") as f:
                json.dump({
                    "method_enc_name": encoder.name,
                    "mean_pose_error": mean_pose_error.tolist(),
                    "std_pose_error": std_pose_error.tolist(),
                    "mean_pose_entropy": mean_pose_entropy.tolist(),
                    "std_pose_entropy": std_pose_entropy.tolist(),
                    "mean_success_rate": np.array(mean_success_rate).tolist(),
                    "all_pose_errors": np.array(pose_errors).tolist(),
                    "all_poses_gt": np.array([cam.cpu().numpy() for cam in cameras]).tolist(),
                    "all_poses_est": np.array(est_cam_poses).tolist(),
                    "all_pose_entropy": np.array(cam_pose_entropy).tolist(),
                }, f, indent=4)     
                
        # output
        output = {
            "mean_pose_error": mean_pose_error,
            "std_pose_error": std_pose_error,
            "mean_pose_entropy": mean_pose_entropy,
            "std_pose_entropy": std_pose_entropy,
            "mean_success_rate": mean_success_rate,
            "all_pose_errors": pose_errors,
            "all_pose_entropy": cam_pose_entropy,
        }
                
        if self.config.run_fine_inversion:
            # execute fine inversion for RF
            output["fine_inversion"] = self._run_rf_fine_inversion(
                encoder=encoder,
                rad_field=rad_field,
                base_rad_field=base_rad_field,
                images=images,
                cameras=cameras,
                init_guess=np.array(est_cam_poses),
                save_results=save_results,
                output_path=output_path,
            )

        return output
        
    def _run_rf_fine_inversion(
        self,
        encoder: BaseImageEncoder,
        rad_field: RFModel,
        images: list[torch.Tensor],
        cameras: list[torch.Tensor],
        base_rad_field: RFModel = None,
        save_results: bool = False,
        output_path: Path | str = "outputs/eval/rf_inversion",
        print_stats: bool = True,
        init_guess: list[torch.Tensor] = None,
        feature_detector: POI_Detector = POI_Detector.SIFT, 
    ):
        # input type-checking
        
        # save results
        if save_results is None:
            save_results = self.config.save_results
            
        # cast to tensor
        if not isinstance(init_guess, torch.Tensor):
            init_guess = torch.tensor(init_guess).float().to(self.device)
         
        # --- Step 1: Get GT camera poses and images ---
        if images is None:
            assert base_rad_field is not None, "A radiance field for image rendering/generation is required!"
            cameras, images = self._load_rf_inversion_data(
                base_rad_field=base_rad_field,
                cameras=cameras,
            )
            
        # --- Step 2: Run the baseline ---
        
        # set encoder 
        encoder = rad_field.pipeline.datamanager.img_semantics_interpolator.model 

        # PnP-RANSAC
        fine_inv_name: str = f"{encoder.name}_fine_inversion"
        
        # update the output path
        output_path: Path = Path(f"{output_path}/{fine_inv_name}")
 
        # make directory, if necessary
        output_path.mkdir(parents=True, exist_ok=True)
         
        # all errors
        pose_errors = []
        est_cam_poses = []
    
        # success flag (i.e., no error thrown)
        success_flags = []
        
        # mean and std error
        mean_pose_error = []
        std_pose_error = []
        
        # mean success rate (i.e., no error thrown)
        mean_success_rate = []
        
        # camera intrinsic params
        _, _, cam_K = rad_field.get_camera_intrinsics() #FIXME: here?
        cam_K = cam_K.to(self.device)
        
        # print
        self.console.rule(f"[bold green]Running fine RF inversion procedure")
            
        for p_idx in tqdm(range(len(images)), 
                            desc="computing error in pose estimates"):
            
            # ground-truth pose
            gt_pose = cameras[p_idx].cpu().numpy()
            
            try:
                # compute the estimated pose
                est_cam_pose = execute_PnP_RANSAC(
                    splat=rad_field,
                    init_guess=init_guess[p_idx],
                    rgb_input=images[p_idx].moveaxis(0, -1),
                    camera_intrinsics_K=cam_K,
                    feature_detector=feature_detector,
                    save_image=False,
                    print_stats=False,
                )
                
                # pose error (rotation error, translation error)
                error = SE3error(gt_pose, est_cam_pose)[:2]
                
                # success flag
                success = 1
            except Exception as excp:
                print(excp)
                
                # est_cam_pose
                est_cam_pose = np.nan * gt_pose
                
                # pose error (rotation error, translation error)
                error = np.nan * np.zeros(2)
                
                # success flag
                success = 0
            
            # append result
            est_cam_poses.append(est_cam_pose)
                
            # cache the error
            pose_errors.append(error)
            
            # cache the success
            success_flags.append(success)
                
            if print_stats and not np.any(np.isnan(error)):
                self.console.log(f"Idx: {p_idx}, SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}")
        
        # --- Step 4: Compute Pose Error Stats ---
        # mean and std error
        mean_pose_error.append(np.nanmean(pose_errors, axis=0))
        std_pose_error.append(np.nanstd(pose_errors, axis=0))

        # mean success rate
        mean_success_rate.append(np.nanmean(success_flags, axis=0))
        
        # --- Step 5: Save results ---
        if save_results:
            with open(f"{output_path}/pose_errors.json", "w") as f:
                json.dump({
                    "method_enc_name": fine_inv_name,
                    "mean_pose_error": np.array(mean_pose_error).tolist(),
                    "std_pose_error": np.array(std_pose_error).tolist(),
                    "mean_success_rate": np.array(mean_success_rate).tolist(),
                    "all_pose_errors": np.array(pose_errors).tolist(),
                    "all_poses_gt": np.array([cam.cpu().numpy() for cam in cameras]).tolist(),
                    "all_poses_est": np.array(est_cam_poses).tolist(),
                }, f, indent=4)        

        return {
            "mean_pose_error": mean_pose_error,
            "std_pose_error": std_pose_error,
            "mean_success_rate": mean_success_rate,
            "all_pose_errors": pose_errors,
        }
    
    def _run_rf_inversion_baselines(
        self,
        rad_field: RFModel,
        images: list[torch.Tensor],
        cameras: list[torch.Tensor],
        base_rad_field: RFModel = None,
        save_results: bool = False,
        output_path: Path | str = "outputs/eval/rf_inversion",
        print_stats: bool = True,
        init_guess: torch.Tensor = None,
        feature_detector: POI_Detector = POI_Detector.SIFT, 
        init_with_perturbed_gt_pose: bool = None,  # perturbs the ground-truth to get an initial guess
        perturb_init_guess_params: dict = None, 
    ):
        # input type-checking
        
        # save results
        if save_results is None:
            save_results = self.config.save_results
            
        # option to perturb the ground-truth for an initial guess
        if init_with_perturbed_gt_pose is None:
            init_with_perturbed_gt_pose = self.config.init_with_perturbed_gt_pose
                 
        # parameters with which to perturb the ground-truth to get the initial guess
        if perturb_init_guess_params is None:
            perturb_init_guess_params = self.config.perturb_init_guess_params
 
        if init_guess is None and not init_with_perturbed_gt_pose:
            raise ValueError(f"An initial guess is required for the camera pose!")
 
        # --- Step 1: Get GT camera poses and images ---
        if images is None:
            assert base_rad_field is not None, "A radiance field for image rendering/generation is required!"
            cameras, images = self._load_rf_inversion_data(
                base_rad_field=base_rad_field,
                cameras=cameras,
            )
            
        # --- Step 2: Run the baseline ---
        
        # iNeRF
        baseline_name: str = "inerf"
        
        # update the output path
        output_path: Path = Path(f"{output_path}/{baseline_name}")
 
        # make directory, if necessary
        output_path.mkdir(parents=True, exist_ok=True)
         
        # all errors
        all_pose_errors = []
        all_est_cam_poses = []
        
        # success flag (i.e., no error thrown)
        all_success_flags = []
        
        # mean and std error
        mean_pose_error = []
        std_pose_error = []
        
        # mean success rate (i.e., no error thrown)
        mean_success_rate = []
        
        # camera intrinsic params
        _, _, cam_K = rad_field.get_camera_intrinsics()
        cam_K = cam_K.to(self.device)

        for ptb_idx in range(len(perturb_init_guess_params["rotation"])):
            # all errors
            pose_errors = []
            est_cam_poses = []
                
           # success flag (i.e., no error thrown)
            success_flags = []
            
            # print
            self.console.rule(f"[bold green]Baseline in eval setting {ptb_idx}")
              
            for p_idx in tqdm(range(len(images)), 
                            desc="computing error in pose estimates"):
                
                # ground-truth pose
                gt_pose = cameras[p_idx].cpu().numpy()
                
                if init_with_perturbed_gt_pose:
                    # set the initial guess to the ground-truth
                    init_guess = gt_pose.copy()
                    
                    # add noise to the initial guess of the pose

                    # generate a random rotation axis
                    rand_rot_axis = torch.nn.functional.normalize(
                        torch.rand(3, device=self.device), dim=-1
                    )

                    # random rotation matrix
                    rand_rot = vec_to_rot_matrix(
                        perturb_init_guess_params["rotation"][ptb_idx] * rand_rot_axis
                    )

                    # initial guess
                    init_guess[:3, :3] = rand_rot.cpu().numpy() @ init_guess[:3, :3]
                    init_guess[:3, 3] += (
                        perturb_init_guess_params["translation"][ptb_idx]
                        * torch.nn.functional.normalize(torch.rand(3, device=self.device), dim=-1)
                        .cpu()
                        .numpy()
                    )
                    init_guess = torch.tensor(init_guess).float().to(self.device)

                try:
                    # compute the estimated pose
                    est_cam_pose = execute_iNeRF(
                        nerf=rad_field,
                        init_guess=init_guess,
                        rgb_input=images[p_idx].moveaxis(0, -1),
                        learning_rate=self.config.rf_inversion_baseline_learning_rate,
                        convergence_threshold=self.config.rf_inversion_baseline_conv_threshold,
                        max_num_iterations=self.config.rf_inversion_baseline_max_num_iterations,
                        batch_size=self.config.rf_inversion_baseline_batch_size,
                        feature_detector=feature_detector,
                        print_stats=False,
                    )
                    
                    # pose error (rotation error, translation error)
                    error = SE3error(gt_pose, est_cam_pose)[:2]
                    
                    # success flag
                    success = 1
                except Exception as excp:
                    print(excp)
                    
                    # est_cam_pose
                    est_cam_pose = np.nan * gt_pose
                    
                    # pose error (rotation error, translation error)
                    error = np.nan * np.zeros(2)
                    
                    # success flag
                    success = 0
                  
                # release memory
                torch.cuda.empty_cache()
                gc.collect()
                            
                # append result
                est_cam_poses.append(est_cam_pose)
                    
                # cache the error
                pose_errors.append(error)
                
                # cache the success
                success_flags.append(success)
                            
                if print_stats and not np.any(np.isnan(error)):
                    self.console.log(f"Idx: {p_idx}, SE(3) Estimation Error -- Rotation: {error[0]}, Translation: {error[1]}")
        
            # append result
            all_pose_errors.append(pose_errors)
            all_est_cam_poses.append(est_cam_poses)
            all_success_flags.append(success_flags)
            
            # --- Step 4: Compute Pose Error Stats ---
            # mean and std error
            mean_pose_error.append(np.nanmean(pose_errors, axis=0))
            std_pose_error.append(np.nanstd(pose_errors, axis=0))
            mean_success_rate.append(np.nanmean(success_flags, axis=0))

        # --- Step 5: Save results ---
        if save_results:
            with open(f"{output_path}/pose_errors.json", "w") as f:
                json.dump({
                    "method_enc_name": baseline_name,
                    "perturb_init_guess_params": perturb_init_guess_params,
                    "mean_pose_error": np.array(mean_pose_error).tolist(),
                    "std_pose_error": np.array(std_pose_error).tolist(),
                    "mean_success_rate": np.array(mean_success_rate).tolist(),
                    "all_pose_errors": np.array(all_pose_errors).tolist(),
                    "all_poses_gt": np.array([cam.cpu().numpy() for cam in cameras]).tolist(),
                    "all_poses_est": np.array(all_est_cam_poses).tolist(),
                    "all_success_flags": np.array(all_success_flags).tolist(),
                }, f, indent=4)        

        return {
            "mean_pose_error": mean_pose_error,
            "std_pose_error": std_pose_error,
            "mean_success_rate": mean_success_rate,
            "all_pose_errors": pose_errors,
        }
        
    @torch.no_grad()
    def _load_rf_inversion_data(
        self,
        base_rad_field: RFModel = None,
        cameras: list[torch.Tensor] = None,
    ):
        """
        Loads the data for radiance field inversion evaluation
        """
        if cameras is None:
            cameras = base_rad_field.pipeline.datamanager.eval_dataset.cameras

        # poses and images
        eval_cam_poses = []
        cam_rgbs = []

        # render
        for cam_idx in tqdm(
            range(len(cameras)),
            desc=f"Loading evaluation cameras",
        ):  
            self.console.rule(f"[bold green]Eval Idx {cam_idx}")
            
            # read pose
            gt_cam =  cameras[cam_idx : cam_idx + 1].to(self.device)
            
            # ground-truth pose
            gt_pose = torch.eye(4, device=self.device)
            gt_pose[:3] = gt_cam.camera_to_worlds
            gt_pose = gt_pose  # .cpu().numpy()
            
            # append
            eval_cam_poses.append(gt_pose)
            
            # query image
            # cam_rgb = eval_dataset[cam_idx]["image"].cuda()
            # cam_rgbs.append(cam_rgb.moveaxis(-1, 0))
            
            # render an RGB image
            rf_outputs = base_rad_field.render(
                cameras=gt_cam,
                compute_semantics=False,
            )
            
            # RGB image
            cam_rgb = rf_outputs["rgb"]
            cam_rgbs.append(cam_rgb.moveaxis(-1, 0))
            
            

        return eval_cam_poses, cam_rgbs