# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch.nn import Parameter

from spine.encoders.image_encoder import BaseImageEncoder
from spine.data.spine_datamanager import SPINEDataManager

from spine.viewer_utils import ViewerUtils
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

# from pytorch3d.transforms import matrix_to_quaternion
from spine.spine_utils import (
    matrix_to_quaternion, 
    _lie_algebra_to_rotmat,
    _rotmat_to_lie_algebra,
    kl_divergence_loss_fn,
    _compute_GMM_entropy,
)

try:
    import tinycudann as tcnn
except ImportError:
    pass

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@torch_compile()
def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


@dataclass
class SPINEModelConfig(ModelConfig):
    """SPINE Model Config, nerfstudio's implementation of Gaussian Splatting"""
    _target: Type = field(default_factory=lambda: SPINEModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    # grid_shape: Tuple[int, int, int] = (16, 16, 8)
    grid_shape: tuple = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""
    strategy: Literal["default", "mcmc"] = "default"
    """The default strategy will be used if strategy is not specified. Other strategies, e.g. mcmc, can be used."""
    max_gs_num: int = 1_000_000
    """Maximum number of GSs. Default to 1_000_000."""
    noise_lr: float = 5e5
    """MCMC samping noise learning rate. Default to 5e5."""
    mcmc_opacity_reg: float = 0.01
    """Regularization term for opacity in MCMC strategy. Only enabled when using MCMC strategy"""
    mcmc_scale_reg: float = 0.01
    """Regularization term for scale in MCMC strategy. Only enabled when using MCMC strategy"""
    enable_semantic_mask: bool = True
    # """Option to utilize the alpha channel as a mask for semantics distillation."""
    semantics_batch_size: int = 4096
    """The batch size for training the semantic field."""
    output_semantics_during_training: bool = False
    """If True, output semantic-scene information during training. Otherwise, only output semantic-scene information during evaluation."""
    semantics_img_loss_weight: float = 1e0
    """weight for the semantics-related term in the loss function."""
    enable_sparsification: bool = False
    """If true, utilizes a sparsity-inducing loss function."""
    sparsity_weight_init: float = 0.0
    """Initial weight for the sparsity-inducing term in the loss function."""
    sparsity_weight_max: float = 2e-4
    """Maximum value of the weight for the sparsity-inducing term in the loss function."""
    sparsity_weight_increment_factor: float = 1 / 2000
    """Increment factor of the weight for the sparsity-inducing term in the loss function."""
    cam_pose_autoencoder_loss_weight_trans: float = 1e0
    """weight for the camera-pose-autoencoder-related term in the loss function."""
    cam_pose_autoencoder_loss_weight_rot: float = 1e0
    """weight for the camera-pose-autoencoder-related term in the loss function."""
    kl_divergence_loss_weight: float = 1e-2
    """weight for the Kullback-Leibler divergence term in the camera-pose-autoencoder-related loss function."""
    use_mse_in_inversion_loss: bool = True
    """option to use the MSE loss as the camera-pose-autoencoder loss function."""
    use_dss_in_inversion_loss: bool = False
    """option to use the Dawid-Sebastiani loss as the camera-pose-autoencoder loss function."""
    use_kl_divergence_component_in_inversion_loss: bool = False
    """weight for the Kullback-Leibler divergence term in the camera-pose-autoencoder-related loss function."""
    output_semantic_pca_during_training: bool = False
    """option to enable PCA visualization of semantic features."""
    base_semantic_dim: int = 64
    """output dimension for the base semantic field."""
    learn_image_to_cam_embedding_pose_encoder: bool = False
    """learn encoder from image to camera embedding"""
    supervise_rf_inverse_map_in_lie_group: bool = True
    """option to supervise the outputs of the camera-to-pose embedding model in so(3) as opposed to SO(3)"""
    num_gmm_components_cam_pose: int = 1
    """Gaussian Mixture Model (GMM) params"""
    num_trans_components: int = 3
    """number of elements in translation parameterization"""
    num_rot_components: int = 3
    """number of elements in rotation parameterization"""
    
    
    # MLP head
    hidden_dim: int = 64 # 128
    num_layers: int = 2
    
    # Positional encoding
    use_pe: bool = True
    pe_n_freq: int = 18
    # Hash grid
    num_levels: int = 12
    log2_hashmap_size: int = 19
    start_res: int = 16
    max_res: int = 128
    features_per_level: int = 8
    hashgrid_layers: tuple = (12, 12)
    hashgrid_resolutions: tuple = ((16, 128), (128, 512))
    hashgrid_sizes: tuple = (19, 19)
    

class SPINEModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SPINEModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):# image encoder
        self.dino_image_encoder: BaseImageEncoder = self.kwargs["dino_image_encoder"]
        self.vggt_image_encoder: BaseImageEncoder = self.kwargs["vggt_image_encoder"]
        self.clip_image_encoder: BaseImageEncoder = self.kwargs["clip_image_encoder"]
        
        # datamanager
        self.datamanager: SPINEDataManager = self.kwargs["datamanager"] 
        
        # output dimension for the base semantic field
        self.base_semantic_field_output_dim: int = self.config.base_semantic_dim
        
        # semantic embeddings input dimension (compatible with a sequence of data features)
        if "cam_feats" in self.kwargs["metadata"]["feature_dim"]:
            self.cam_semantic_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]["cam_feats"]
        else:
            self.cam_semantic_embeds_input_dim = 0
            
        if "img_feats" in self.kwargs["metadata"]["feature_dim"]:
            self.img_semantic_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]["img_feats"]
        else:
            self.img_semantic_embeds_input_dim = 0
            
        # TODO: add support for augmented image semantic feature distillation
        if "aug_feats" in self.kwargs["metadata"]["feature_dim"]:
            self.aug_semantic_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]["aug_feats"]
        else:
            self.aug_semantic_embeds_input_dim = 0
        
        # vision-language semantics
        if "lang_img_feats" in self.kwargs["metadata"]["feature_dim"]:
            self.lang_semantic_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]["lang_img_feats"]
        else:
            self.lang_semantic_embeds_input_dim = 0
            
        # dimension for the first block in the encoder
        self.semantic_embeds_block_0_dim = 128
        
        # distillation options
        self.camera_pose_estimation_enabled = self.kwargs["metadata"]["camera_pose_estimation_enabled"]
        self.img_semantic_feature_distillation_enabled = self.kwargs["metadata"]["img_semantic_feature_distillation_enabled"]
        self.lang_semantic_feature_distillation_enabled = self.kwargs["metadata"]["lang_semantic_feature_distillation_enabled"]
        
        # dimension of the estimated camera pose
        self.dim_est_cam_pose: int = self.config.num_trans_components + self.config.num_rot_components
    
        # dimension of the parameters for the latent distributions
        self.dim_cam_pose_distribution_params = (
            self.config.num_gmm_components_cam_pose * self.dim_est_cam_pose * 2  # GMM with mean and variance
        )
            
        # parameters for the discrete distribution over the latents (cam pose)
        self.dim_cam_pose_latent_distribution = self.config.num_gmm_components_cam_pose
        
        # dimension of the estimated camera pose
        cam_pose_output_dim = self.dim_cam_pose_latent_distribution + self.dim_cam_pose_distribution_params
        
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        
        # semantic field
        growth_factor = np.exp(
            (np.log(self.config.max_res) - np.log(self.config.start_res)) 
            / (self.config.num_levels - 1)
        )

        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": self.config.num_levels,
                    "n_features_per_level": self.config.features_per_level,
                    "log2_hashmap_size": self.config.log2_hashmap_size,
                    "base_resolution": self.config.start_res,
                    "per_level_scale": growth_factor,
                }
            ],
        }

        if self.config.use_pe:
            encoding_config["nested"].append(
                {
                    "otype": "Frequency",
                    "n_frequencies": self.config.pe_n_freq,
                    "n_dims_to_encode": 3,
                }
            )
            
        # base semantic field
        if self.img_semantic_feature_distillation_enabled or self.lang_semantic_feature_distillation_enabled:
            self.base_semantic_field = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=self.base_semantic_field_output_dim,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_dim,
                    "n_hidden_layers": self.config.num_layers,
                },
            )
            
        if self.img_semantic_feature_distillation_enabled:
            self.img_semantic_field = tcnn.Network(
                n_input_dims=self.base_semantic_field_output_dim,
                n_output_dims=self.img_semantic_embeds_input_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_dim,
                    "n_hidden_layers": self.config.num_layers,
                },
            )
            
        # vision-language semantic field
        if self.lang_semantic_feature_distillation_enabled:
            self.lang_semantic_field = tcnn.Network(
                n_input_dims=self.base_semantic_field_output_dim,
                n_output_dims=self.lang_semantic_embeds_input_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_dim,
                    "n_hidden_layers": self.config.num_layers,
                },
            )
          
        # inversion field (img-to-pose)
        if self.camera_pose_estimation_enabled:

            if self.config.learn_image_to_cam_embedding_pose_encoder:
                encoding_config = {
                    "otype": "Composite",
                    "nested": [
                        {
                            "otype": "HashGrid",
                            "n_levels": self.config.num_levels,
                            "n_features_per_level": self.config.features_per_level,
                            "log2_hashmap_size": self.config.log2_hashmap_size,
                            "base_resolution": self.config.start_res,
                            "per_level_scale": growth_factor,
                        }
                    ],
                }
                
                # camera pose to semantic embedding (encoder)
                self.cam_pose_to_embedding_quats = tcnn.NetworkWithInputEncoding(
                    n_input_dims=4,
                    n_output_dims=self.semantic_embeds_block_0_dim,
                    encoding_config=encoding_config,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": self.config.hidden_dim // 4,
                        "n_hidden_layers": self.config.num_layers,
                    },
                )

                if self.config.use_pe:
                    encoding_config["nested"].append(
                        {
                            "otype": "Frequency",
                            "n_frequencies": self.config.pe_n_freq,
                            "n_dims_to_encode": 3,
                        }
                    )
                    
                
                # camera pose to semantic embedding (encoder)
                self.cam_pose_to_embedding_trans = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3,
                    n_output_dims=self.semantic_embeds_block_0_dim,
                    encoding_config=encoding_config,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": self.config.hidden_dim // 4,
                        "n_hidden_layers": self.config.num_layers,
                    },
                )
                
                self.cam_pose_to_embedding = tcnn.Network(
                    n_input_dims=self.semantic_embeds_block_0_dim * 2,
                    n_output_dims=self.cam_semantic_embeds_input_dim,
                    # encoding_config=encoding_config,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": self.config.hidden_dim // 2,
                        "n_hidden_layers": self.config.num_layers,
                    },
                )
            
            # semantic embedding to camera pose (decoder)
            self.embedding_to_cam_pose = tcnn.Network(
                n_input_dims=self.cam_semantic_embeds_input_dim,
                n_output_dims=cam_pose_output_dim,
                # encoding_config=encoding_config,
                network_config={
                    # "otype": "FullyFusedMLP",
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_dim // 2,
                    "n_hidden_layers": self.config.num_layers,
                },
            )
                
        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        if self.config.strategy == "default":
            # Strategy for GS densification
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(f"""Splatfacto does not support strategy {self.config.strategy} 
                             Currently, the supported strategies include default and mcmc.""")

        if self.config.enable_sparsification:
            # the weight on the sparsity-inducing component in the loss function
            self.sparsity_weight = self.config.sparsity_weight_init
            
            # difference between the initial value and the maximum value of the wieght of the sparsity-inducing component
            self.sparsity_weight_diff = self.config.sparsity_weight_max - self.config.sparsity_weight_init

        # initialize Viewer
        self.viewer_utils = ViewerUtils(self.clip_image_encoder)

        self.setup_gui()

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        else:
            return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)
        
    def setup_gui(self):
        self.viewer_utils.device = "cuda:0"  # self.device
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton(
            "Refresh PCA Projection",
            cb_hook=lambda _: self.viewer_utils.reset_pca_proj(),
        )

        # Only setup GUI for language features if we're using CLIP
        self.hint_text = ViewerText(
            name="Note:", disabled=True, default_value="Use , to separate labels"
        )
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(
                elem.value, is_positive=True
            ),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(
                elem.value, is_positive=False
            ),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=self.viewer_utils.softmax_temp,
            cb_hook=lambda elem: self.viewer_utils.update_softmax_temp(elem.value),
        )
        
    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],  # the learning rate for the "means" attribute of the GS
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        return cbs

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        
        # insert parameters for the semantic fields
        # inversion field
        if self.camera_pose_estimation_enabled:
            if self.config.learn_image_to_cam_embedding_pose_encoder:
                gps["cam_pose_to_embedding"] = (
                    list(self.cam_pose_to_embedding_trans.parameters())
                    + list(self.cam_pose_to_embedding_quats.parameters())
                    + list(self.cam_pose_to_embedding.parameters())
                )
            gps["embedding_to_cam_pose"] = list(self.embedding_to_cam_pose.parameters())
            
        # image-space semantics
        if self.img_semantic_feature_distillation_enabled:
            gps["img_semantic_field"] = list(self.img_semantic_field.parameters())
        
        # vision-language semantics
        if self.lang_semantic_feature_distillation_enabled:
            gps["lang_semantic_field"] = list(self.lang_semantic_field.parameters())
        
        # base semantics
        if self.img_semantic_feature_distillation_enabled or self.lang_semantic_feature_distillation_enabled:
            gps["base_semantic_field"] = list(self.base_semantic_field.parameters())
        
        # camera optimizer
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]
    
    @torch_compile()
    def encode_camera_pose(self, camera_pose: torch.Tensor) -> torch.Tensor:
        """Takes in a Camera pose and encodes it in a latent space.

        Args:
            camera_pose: Camera pose

        Returns:
            latent embedding of the camera pose and the associated camera pose (translation and quaternion)
        """
        # get translation and quaternions from camera2world
        camera_pose = camera_pose.detach()
        cam_trans = camera_pose[:, :3, 3:4].squeeze(dim=-1)
        
        # rotation matrix
        cam_rot = camera_pose[:, :3, :3].detach()
        
        # compute the latent embeddings
        cam_pose_enc = None
        
        if self.config.learn_image_to_cam_embedding_pose_encoder:
            # TODO: add support
            cam_pose_enc = None
        
        return cam_pose_enc, cam_trans, cam_rot
    
    def decode_camera_pose(
        self, 
        latent_camera_embeds: torch.Tensor,
        use_expected_value: bool=False,
        compute_uncertainty: bool = False,
    ) -> torch.Tensor:
        """Takes in a camera latent embeddings and decodes to a camera2world pose.

        Args:
            latent_camera_embeds: camera latent embeddings

        Returns:
            camera2world pose.
        """
        
        # decode the latent embeddings
        cam_pose_dec = self.embedding_to_cam_pose(latent_camera_embeds)  
        
        # split into latent distribution and the Gaussian components
        cam_pose_latent_distrib = cam_pose_dec[..., :self.dim_cam_pose_latent_distribution]
        cam_pose_gmm_comp = cam_pose_dec[..., self.dim_cam_pose_latent_distribution:]
        
        # starting index for the logvar components (wihin cam_pose_gmm_comp)
        start_idx_logvar = self.config.num_gmm_components_cam_pose * self.dim_est_cam_pose
        
        # take the softmax and compute the argmax over the latents
        cam_pose_latent_softmax = torch.softmax(cam_pose_latent_distrib, dim=-1)      
        
        # get the argmax to select a component
        pose_latent_argmax = torch.argmax(cam_pose_latent_softmax, dim=0)
        
        # retrieve the mean and variance of the Gaussian component
        pose_mean =  cam_pose_gmm_comp[...,
                                       pose_latent_argmax * self.dim_est_cam_pose :
                                       (pose_latent_argmax + 1) * self.dim_est_cam_pose]
        pose_logvar =  cam_pose_gmm_comp[...,
                                         start_idx_logvar 
                                         + pose_latent_argmax * self.dim_est_cam_pose :
                                         start_idx_logvar 
                                         + (pose_latent_argmax + 1) * self.dim_est_cam_pose]

        # sample from the Gaussian component
        if not use_expected_value:
            cam_pose = pose_mean + torch.exp(pose_logvar) * torch.normal(
                mean=torch.zeros(pose_mean[:, 0:1].shape), std=torch.ones(pose_mean[:, 0:1].shape)
            ).to(pose_logvar.device)
        else:
            cam_pose = pose_mean
        
        # camera translation and rotation components
        cam_pose_trans, cam_pose_rot = (
            cam_pose[..., :self.config.num_trans_components],
            cam_pose[..., self.config.num_rot_components:],
        )
        
        # uncertainty (entropy) of the estimate
        entropy = None
            
        if compute_uncertainty:
            # log-variance for the GMM
            distrib_logvar =  cam_pose_gmm_comp[...,
                                                start_idx_logvar :
                                                start_idx_logvar 
                                                + self.config.num_gmm_components_cam_pose * self.dim_est_cam_pose
                                               ]
            
            # reshape into (num_components, pose dimension)
            distrib_logvar = distrib_logvar.reshape(-1, self.config.num_gmm_components_cam_pose, self.dim_est_cam_pose)
            
            # weights for the GMM
            weights = cam_pose_latent_softmax
            
            # compute the uncertainty (entropy) of the estimate
            entropy = _compute_GMM_entropy(
                weights=weights,
                cov=torch.exp(distrib_logvar),
            )
            
        # output
        output = {
            "pose_trans": cam_pose_trans,
            "pose_rot": cam_pose_rot,
            "pose_mean": pose_mean,
            "pose_logvar": pose_logvar,
            "pose_mean_trans": pose_mean[..., :self.config.num_trans_components],
            "pose_mean_rot": pose_mean[..., self.config.num_rot_components:],
            "pose_logvar_trans": pose_logvar[..., :self.config.num_trans_components],
            "pose_logvar_rot": pose_logvar[..., self.config.num_rot_components:],
            "entropy": entropy,
        }
        
        return output
        
        
    # @torch.compile()
    @torch.no_grad()
    def compute_semantic_pca(self, semantic_embeds: torch.Tensor, num_pca_components: int = 3) -> torch.Tensor:
        """Takes in a semantic embedding and computes the PCA"""
        # flattened semantic features
        sem_features_rs = semantic_embeds.reshape(-1, semantic_embeds.shape[-1]).float()
        
        # PCA
        U_pca_lr, S_pca_lr, V_pca_lr = torch.pca_lowrank(
            A=sem_features_rs,
            q=num_pca_components,
            center=True, # default
            niter=2, # default
        )
        
        # project to the first-3 principal directions
        semantic_img_proj = sem_features_rs @ V_pca_lr[:, :num_pca_components]
        
        # extract background and foreground
        semantic_img_proj_bgd = semantic_img_proj[..., 0:1].detach().clone()
        semantic_img_proj = semantic_img_proj[..., 0:3]
        
        # map to the image space in [0, 1]
        semantic_img_proj -= semantic_img_proj.min(dim=0, keepdim=True)[0]
        semantic_img_proj /= semantic_img_proj.max(dim=0, keepdim=True)[0]
        
        # extract the source and target images
        semantic_img_proj = (
            semantic_img_proj.reshape(*semantic_embeds.shape[:-1], 3) # num_pca_components - 1)
        )
        
        # background
        semantic_img_proj_bgd = (
            semantic_img_proj_bgd.reshape(*semantic_embeds.shape[:-1], 1)
        )
        
        return semantic_img_proj, semantic_img_proj_bgd
        
    # @torch.no_grad()
    def get_semantic_outputs(self, outputs: Dict[str, torch.Tensor]):
        if not self.training:
            # Normalize CLIP features rendered by feature field
            clip_features = outputs["lang_semantics"]
            clip_features /= clip_features.norm(dim=-1, keepdim=True)

            if self.viewer_utils.has_positives:
                if self.viewer_utils.has_negatives:
                    # Use paired softmax method as described in the paper with positive and negative texts
                    text_embs = torch.cat(
                        [self.viewer_utils.pos_embed, self.viewer_utils.neg_embed],
                        dim=0,
                    )

                    raw_sims = clip_features @ text_embs.T

                    # Broadcast positive label similarities to all negative labels
                    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
                    pos_sims = pos_sims.broadcast_to(neg_sims.shape)

                    # Updated Code
                    paired_sims = torch.cat(
                        (pos_sims.reshape((-1, 1)), neg_sims.reshape((-1, 1))), dim=-1
                    )

                    # compute the paired softmax
                    probs = paired_sims.softmax(dim=-1)[..., :1]
                    probs = probs.reshape((-1, neg_sims.shape[-1]))

                    torch.nan_to_num_(probs, nan=0.0)

                    sims, _ = probs.min(dim=-1, keepdim=True)
                    outputs["lang_similarity"] = sims.reshape((*pos_sims.shape[:-1], 1))

                    # cosine similarity
                    outputs["lang_raw_similarity"] = raw_sims[..., :1]
                else:
                    # positive embeddings
                    text_embs = self.viewer_utils.pos_embed

                    sims = clip_features @ text_embs.T
                    # Show the mean similarity if there are multiple positives
                    if sims.shape[-1] > 1:
                        sims = sims.mean(dim=-1, keepdim=True)
                    outputs["lang_similarity"] = sims

                    # cosine similarity
                    outputs["lang_raw_similarity"] = sims

                # for outputs similar to the GUI
                similarity_clip = outputs[f"lang_similarity"] - outputs[f"lang_similarity"].min()
                similarity_clip /= similarity_clip.max() + 1e-10
                outputs["lang_similarity_GUI"] = apply_colormap(
                    similarity_clip, ColormapOptions("turbo")
                )
                
            if "rgb" in outputs.keys():
                if self.viewer_utils.has_positives:
                    # composited similarity
                    p_i = torch.clip(outputs["lang_similarity"] - 0.5, 0, 1)

                    outputs["lang_composited_similarity"] = apply_colormap(
                        p_i / (p_i.max() + 1e-6), ColormapOptions("turbo")
                    )
                    mask = (outputs["lang_similarity"] < 0.5).squeeze()
                    outputs["lang_composited_similarity"][mask, :] = outputs["rgb"][mask, :]

        return outputs

    def get_point_cloud_from_camera(self,
                                    camera: Cameras,
                                    depth: torch.Tensor,
                                    ) -> torch.Tensor:
        """Takes in a Camera and returns the back-projected points.

        Args:
            camera: Input Camera. This Camera Object should have all the
            needed information to compute the back-projected points.
            depth: Predicted depth image.

        Returns:
            back-projected points from the camera.
        """
        # camera intrinsics
        H, W, K = camera.height.item(), camera.width.item(), camera.get_intrinsics_matrices()
        K = K.squeeze()
        
        # unnormalized pixel coordinates
        u_coords = torch.arange(W, device=self.device)
        v_coords = torch.arange(H, device=self.device)

        # meshgrid
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')

        # transformed points in camera frame
        # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
        cam_pts_x = (U_grid - K[0, 2]) * depth.squeeze() / K[0, 0]
        cam_pts_y = (V_grid - K[1, 2]) * depth.squeeze() / K[1, 1]
        
        cam_pcd_points = torch.stack((cam_pts_x, cam_pts_y,
                                        depth.squeeze(), 
                                        torch.ones_like(cam_pts_y)),
                                        axis=-1).to(self.device)
        
        # camera pose
        cam_pose = torch.eye(4, device=self.device)
        cam_pose[:3] = camera.camera_to_worlds
        
        # convert from OpenGL to OpenCV Convention
        cam_pose[:, 1] = -cam_pose[:, 1]
        cam_pose[:, 2] = -cam_pose[:, 2]
        
        # point = torch.einsum('ij,hkj->hki', cam_pose, cam_pcd_points)
        
        point = cam_pose @ cam_pcd_points.view(-1, 4).T
        point = point.T.view(*cam_pcd_points.shape[:2], 4)
        point = point[..., :3].view(*depth.shape[:2], 3)
        
        return point

    def get_outputs(self, camera: Cameras,
                    compute_semantics: Optional[bool] = True) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # generate a point cloud from the depth image
        pcd_points = self.get_point_cloud_from_camera(camera, depth_im.detach().clone())
        
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        
        # selected indices and points
        sel_idx = None
        
        # predicted image-space semantic embeddings
        img_semantic_im = None
        
        # predicted vision-language semantic embeddings
        lang_semantic_im = None
        
        # predict semantics for the entire image, if visualization is specified
        # number of points to subsample
        n_sub_sample = (
            self.config.semantics_batch_size 
            if not self.config.output_semantic_pca_during_training
            else pcd_points.view(-1, 3).shape[0]
        ) 
        
        if self.training and \
        (
            self.img_semantic_feature_distillation_enabled or 
            self.lang_semantic_feature_distillation_enabled
        ):
            # subsample the points
            # n_sub_sample = pcd_points.view(-1, 3).shape[0]
            
            # get random samples
            sel_idx = torch.randperm(pcd_points.view(-1, 3).shape[0],
                                     device=self.device)[:n_sub_sample]
            
            # selected points
            sel_pcd_points = pcd_points.view(-1, 3)[sel_idx]
            
            # predicted base semantic embeddings
            base_sem_im = self.base_semantic_field(sel_pcd_points).float()
            
            if self.img_semantic_feature_distillation_enabled:
                # predicted image-space semantic embeddings
                img_semantic_im = self.img_semantic_field(base_sem_im).float()
            
            if self.lang_semantic_feature_distillation_enabled:
                # predicted vision-language semantic embeddings
                lang_semantic_im = self.lang_semantic_field(base_sem_im).float()
        elif compute_semantics and \
        (
            self.img_semantic_feature_distillation_enabled or 
            self.lang_semantic_feature_distillation_enabled
        ):
            # predicted base semantic embeddings
            base_sem_im = self.base_semantic_field(pcd_points.view(-1, 3)).float()
            
            if self.img_semantic_feature_distillation_enabled:
                # predicted image-space semantic embeddings
                img_semantic_im = self.img_semantic_field(base_sem_im).view(
                    *depth_im.shape[:2], self.img_semantic_embeds_input_dim
                ).float()
            
            if self.lang_semantic_feature_distillation_enabled:
                # predicted vision-language semantic embeddings
                lang_semantic_im = self.lang_semantic_field(base_sem_im).view(
                    *depth_im.shape[:2], self.lang_semantic_embeds_input_dim
                ).float()
            
        # semantic PCA
        semantic_im_pca = None
        
        # TODO: Needs a speed boost
        # compute the semantic PCA
        if not self.training and img_semantic_im is not None:
            semantic_im_pca, _ = self.compute_semantic_pca(img_semantic_im.detach())
            semantic_im_pca = semantic_im_pca.view(
                *depth_im.shape[:2], 3
            ).float()
          
        # outputs
        outputs = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "sel_idx": sel_idx,  # type: ignore
            "img_semantics": img_semantic_im,  # type: ignore
            "lang_semantics": lang_semantic_im,  # type: ignore
            "semantics_pca": semantic_im_pca,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "optimized_camera_to_world": optimized_camera_to_world,  # type: ignore
        }  # type: ignore

        if (
            self.config.output_semantics_during_training or not self.training
        ) and compute_semantics:
            # Compute semantic inputs, e.g., composited similarity.
            outputs = self.get_semantic_outputs(outputs=outputs)

        return outputs

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
            
        # loss: semantic Embeddings
        semantic_img_loss = 0.0
                
        # image-space and vision-language semantics
        all_pred_semantics = []
        all_batch_semantics = []
        
        if self.img_semantic_feature_distillation_enabled:
            # predicted image-space semantic embeddings
            all_pred_semantics.append(outputs["img_semantics"])
            all_batch_semantics.append(batch["img_semantics"][1])
        
        if self.lang_semantic_feature_distillation_enabled:
            # predicted vision-language semantic embeddings
            all_pred_semantics.append(outputs["lang_semantics"])
            all_batch_semantics.append(batch["lang_semantics"][1])
    
        for pred_semantics, batch_semantics in zip(all_pred_semantics, all_batch_semantics):
            if outputs["sel_idx"] is not None:
                # convert linear indices to row-column indices
                sel_idx_row, sel_idx_col = outputs["sel_idx"] // outputs["rgb"].shape[1], outputs["sel_idx"] % outputs["rgb"].shape[1]
                
                # scale factors
                scale_h = batch_semantics.shape[0] / outputs["rgb"].shape[0]
                scale_w = batch_semantics.shape[1] / outputs["rgb"].shape[1]
                
                # scaled indices
                sc_y_ind = (sel_idx_row * scale_h).long()
                sc_x_ind = (sel_idx_col * scale_w).long()
                    
                # ground-truth semantic embeddings
                gt_semantics = batch_semantics[sc_y_ind, sc_x_ind, :].float()
                
                # mask using the alpha channel
                if self.config.enable_semantic_mask:
                    if batch["image"].shape[-1] > 3:
                        # mask
                        mask = self.get_gt_img(batch["image"])[..., -1].float()[..., None].reshape(-1, 1)[outputs["sel_idx"]]
                        pred_semantics = pred_semantics * mask
                        gt_semantics = gt_semantics * mask
                                
                # loss: semantic embeddings
                semantic_img_loss += self.config.semantics_img_loss_weight * (
                    torch.nn.functional.mse_loss(
                        pred_semantics, 
                        gt_semantics,
                    ) 
                    + 
                    (1 - torch.nn.functional.cosine_similarity(
                        pred_semantics, 
                        gt_semantics,
                        dim=-1,
                        )
                        ).mean()
                    )
        
        # autoencoder loss for radiance field inversion
        cam_pose_autoencoder_loss = 0.0
            
        if self.camera_pose_estimation_enabled:
            # ground-truth latent embeddings
            gt_cam_latents = batch["img_semantics"][0].half()[None]
            gt_cam_latents = F.normalize(gt_cam_latents, p=2, dim=-1)
            
            # encode the camera pose
            cam_pose_enc, cam_trans, cam_rot = self.encode_camera_pose(outputs["optimized_camera_to_world"])
            
            # decode the camera latent embeddings
            cam_pose_dec = None
            
            if self.config.learn_image_to_cam_embedding_pose_encoder:
                cam_pose_dec = self.decode_camera_pose(cam_pose_enc)
             
            # end-to-end supervision or teacher forcing
            cam_pose_dec_output = self.decode_camera_pose(
                cam_pose_dec if cam_pose_dec is not None else gt_cam_latents
            )
            
            # unpack decoder output
            cam_pose_dec_trans = cam_pose_dec_output["pose_trans"]
            cam_pose_dec_rot = cam_pose_dec_output["pose_rot"]
            cam_pose_mean = cam_pose_dec_output["pose_mean"]
            cam_pose_logvar = cam_pose_dec_output["pose_logvar"]
            cam_pose_entropy = cam_pose_dec_output["entropy"]
            
            if self.config.supervise_rf_inverse_map_in_lie_group:
                # map ground-truth rotation to lie-algebra
                cam_rot = _rotmat_to_lie_algebra(cam_rot)
            else:
                # map lie algebra to rotation matrix
                cam_pose_dec_rot = _lie_algebra_to_rotmat(cam_pose_dec_rot)
                
            # squeeze the estimates and ground-truth
            cam_pose_dec_trans = cam_pose_dec_trans.half().squeeze()
            cam_pose_dec_rot = cam_pose_dec_rot.half().squeeze()
            cam_trans = cam_trans.half().squeeze()
            cam_rot = cam_rot.half().squeeze()
                
            # autoencoder loss
            if self.config.use_dss_in_inversion_loss:
                # use a uniform loss weight
                cam_pose_loss_weight = (
                    self.config.cam_pose_autoencoder_loss_weight_trans
                    + self.config.cam_pose_autoencoder_loss_weight_rot
                ) / 2.0
                
                # ground-truth camera pose
                gt_cam_pose = torch.concat((cam_trans, cam_rot))
                
                # autoencoder loss
                cam_pose_autoencoder_loss = cam_pose_loss_weight * (
                    2 * cam_pose_logvar 
                    + (cam_pose_mean - gt_cam_pose) / torch.clamp(torch.exp(2 * cam_pose_logvar), min=1e-5)
                ).nanmean()
            elif self.config.use_mse_in_inversion_loss:
                # autoencoder loss
                cam_pose_autoencoder_loss = (
                    self.config.cam_pose_autoencoder_loss_weight_trans * torch.nn.functional.mse_loss(
                        cam_pose_dec_trans,
                        cam_trans,
                    )  # translation
                    + 
                    self.config.cam_pose_autoencoder_loss_weight_rot * torch.nn.functional.mse_loss(
                        cam_pose_dec_rot,
                        cam_rot,
                    )  # rotation
                )
            else:
                raise RuntimeError("Please specify a loss function for training the inverse RF!")
            
            if self.config.use_kl_divergence_component_in_inversion_loss:
                cam_pose_autoencoder_loss += self.config. kl_divergence_loss_weight * kl_divergence_loss_fn(
                    cam_pose_mean, cam_pose_logvar
                )
            
            if self.config.learn_image_to_cam_embedding_pose_encoder:
                cam_pose_autoencoder_loss += self.config.cam_pose_autoencoder_loss_weight * (
                    torch.nn.functional.mse_loss(cam_pose_enc, gt_cam_latents)
                )
             
            if torch.any(torch.isnan(cam_pose_autoencoder_loss)):
                print("NaN detected!")
                breakpoint()
            
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
            
        # main loss
        main_loss = (
            (1 - self.config.ssim_lambda) * Ll1 
            + self.config.ssim_lambda * simloss
            + semantic_img_loss
            + cam_pose_autoencoder_loss
        )
              
        if self.config.enable_sparsification:
            # sparsity-inducing loss
            sparsity_loss = torch.abs(torch.sigmoid(self.opacities)).mean()
            
            # weight of the sparsity-inducing component
            self.sparsity_weight = torch.minimum(torch.tensor(self.sparsity_weight + self.config.sparsity_weight_increment_factor * self.sparsity_weight_diff),
                                                 torch.tensor(self.config.sparsity_weight_max)).to(self.device)

            # loss function
            main_loss += (self.sparsity_weight * sparsity_loss)    
                
        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
        }

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(
        self, 
        camera: Cameras,
        obb_box: Optional[OrientedBox] = None,
        compute_semantics: Optional[bool] = True,
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
            compute_semantics: option to compute semantics
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), compute_semantics=compute_semantics)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
