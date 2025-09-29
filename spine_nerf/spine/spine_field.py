from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from spine.spine_fieldheadnames import SPINEFieldHeadNames
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.misc import torch_compile

from spine.spine_utils import _compute_GMM_entropy

try:
    import tinycudann as tcnn
except ImportError:
    pass

        
class SPINEField(Field):
    def __init__(
        self,
        spatial_distortion: SpatialDistortion,
        # Positional encoding
        use_pe: bool = True,
        pe_n_freq: int = 6,
        # Hash grid
        num_levels: int = 12,
        log2_hashmap_size: int = 19,
        start_res: int = 16,
        max_res: int = 128,
        features_per_level: int = 8,
        # MLP head
        hidden_dim: int = 64,
        num_layers: int = 2,
        # semantic fields
        base_semantic_field_output_dim: int = None,
        img_semantic_embeds_input_dim: int = None,
        lang_semantic_embeds_input_dim: int = None,
        cam_semantic_embeds_input_dim: int = None,
        # distillation options
        img_semantic_feature_distillation_enabled: bool = True,
        lang_semantic_feature_distillation_enabled: bool = True,
        camera_pose_estimation_enabled: bool = True,
        learn_image_to_cam_embedding_pose_encoder: bool = False,
        # inverse map
        dim_est_cam_pose: int = None,
        dim_cam_pose_distribution_params: int = None,
        dim_cam_pose_latent_distribution: int = None,
        cam_pose_output_dim: int = None,
        num_gmm_components_cam_pose: int = None,
        num_trans_components: int = None,
        num_rot_components: int = None,
    ):
        super().__init__()
        
        # Positional encoding
        self.use_pe: bool = use_pe
        self.pe_n_freq: int = pe_n_freq
        # Hash grid
        self.num_levels: int = num_levels
        self.log2_hashmap_size: int = log2_hashmap_size
        self.start_res: int = start_res
        self.max_res: int = max_res
        self.features_per_level: int = features_per_level
        # MLP head
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        
        # semantic dimensions
        self.base_semantic_field_output_dim: int = base_semantic_field_output_dim
        self.img_semantic_embeds_input_dim: int = img_semantic_embeds_input_dim
        self.lang_semantic_embeds_input_dim: int = lang_semantic_embeds_input_dim
        self.cam_semantic_embeds_input_dim: int = cam_semantic_embeds_input_dim
        
        # inverse map dimensions
        # dimension of the estimated camera pose
        self.dim_est_cam_pose: int = dim_est_cam_pose
    
        # dimension of the parameters for the latent distributions
        self.dim_cam_pose_distribution_params = dim_cam_pose_distribution_params
            
        # parameters for the discrete distribution over the latents (cam pose)
        self.dim_cam_pose_latent_distribution = dim_cam_pose_latent_distribution
        
        # dimension of the estimated camera pose
        self.cam_pose_output_dim: int = cam_pose_output_dim
        
        # number of Gaussians in GMM for inverse map
        self.num_gmm_components_cam_pose: int = num_gmm_components_cam_pose
        
        # dimension of the translation and rotation representations
        self.num_trans_components: int = num_trans_components
        self.num_rot_components: int = num_rot_components
        
        # distillation options
        self.img_semantic_feature_distillation_enabled: bool = img_semantic_feature_distillation_enabled
        self.lang_semantic_feature_distillation_enabled: bool = lang_semantic_feature_distillation_enabled
        self.camera_pose_estimation_enabled: bool = camera_pose_estimation_enabled
        self.learn_image_to_cam_embedding_pose_encoder: bool = learn_image_to_cam_embedding_pose_encoder
        
        # spatial distortion
        self.spatial_distortion = spatial_distortion
        
        # dimension for the first block in the encoder
        self.semantic_embeds_block_0_dim = 128
        
       # semantic field
        growth_factor = np.exp(
            (np.log(self.max_res) - np.log(self.start_res)) 
            / (self.num_levels - 1)
        )

        encoding_config = {
            "otype": "Composite",
            "nested": [
                {
                    "otype": "HashGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.start_res,
                    "per_level_scale": growth_factor,
                }
            ],
        }

        if self.use_pe:
            encoding_config["nested"].append(
                {
                    "otype": "Frequency",
                    "n_frequencies": self.pe_n_freq,
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
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers,
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
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers,
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
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers,
                },
            )
          
        # inversion field (img-to-pose)
        if self.camera_pose_estimation_enabled:

            if self.learn_image_to_cam_embedding_pose_encoder:
                encoding_config = {
                    "otype": "Composite",
                    "nested": [
                        {
                            "otype": "HashGrid",
                            "n_levels": self.num_levels,
                            "n_features_per_level": self.features_per_level,
                            "log2_hashmap_size": self.log2_hashmap_size,
                            "base_resolution": self.start_res,
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
                        "n_neurons": self.hidden_dim // 4,
                        "n_hidden_layers": self.num_layers,
                    },
                )

                if self.config.use_pe:
                    encoding_config["nested"].append(
                        {
                            "otype": "Frequency",
                            "n_frequencies": self.pe_n_freq,
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
                        "n_neurons": self.hidden_dim // 4,
                        "n_hidden_layers": self.num_layers,
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
                        "n_neurons": self.hidden_dim // 2,
                        "n_hidden_layers": self.num_layers,
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
                    "n_neurons": self.hidden_dim // 2,
                    "n_hidden_layers": self.num_layers,
                },
            )
    
    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc
    
    # def get_density(
    #     self, ray_samples: RaySamples
    # ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
    #     raise NotImplementedError("get_density not supported for FeatureField")

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        # Apply scene contraction
        outputs = {}
        positions = ray_samples.frustums.get_positions().detach()
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        positions_flat = positions.view(-1, 3)

        # Get features
        if (
            self.img_semantic_feature_distillation_enabled or 
            self.lang_semantic_feature_distillation_enabled
        ):
            # predicted base semantic embeddings
            base_sem_im = self.base_semantic_field(positions_flat)

            # image-space semantic features
            img_semantic_feats = None
            
            if self.img_semantic_feature_distillation_enabled:
                img_semantic_feats = self.img_semantic_field(base_sem_im).view(*ray_samples.frustums.directions.shape[:-1], -1)
            outputs[SPINEFieldHeadNames.IMG_SEMANTICS] = img_semantic_feats
            
            # vision-language semantics features
            lang_semantic_feats = None
            
            if self.lang_semantic_feature_distillation_enabled:
                lang_semantic_feats = self.lang_semantic_field(base_sem_im).view(*ray_samples.frustums.directions.shape[:-1], -1)
            outputs[SPINEFieldHeadNames.LANG_SEMANTICS] = lang_semantic_feats
        
        return outputs
        
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
        
        if self.learn_image_to_cam_embedding_pose_encoder:
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
        start_idx_logvar = self.num_gmm_components_cam_pose * self.dim_est_cam_pose
        
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
            cam_pose[..., :self.num_trans_components],
            cam_pose[..., self.num_rot_components:],
        )
        
        # uncertainty (entropy) of the estimate
        entropy = None
            
        if compute_uncertainty:
            # log-variance for the GMM
            distrib_logvar =  cam_pose_gmm_comp[...,
                                                start_idx_logvar :
                                                start_idx_logvar 
                                                + self.num_gmm_components_cam_pose * self.dim_est_cam_pose
                                               ]
            
            # reshape into (num_components, pose dimension)
            distrib_logvar = distrib_logvar.reshape(-1, self.num_gmm_components_cam_pose, self.dim_est_cam_pose)
            
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
            "pose_mean_trans": pose_mean[..., :self.num_trans_components],
            "pose_mean_rot": pose_mean[..., self.num_rot_components:],
            "pose_logvar_trans": pose_logvar[..., :self.num_trans_components],
            "pose_logvar_rot": pose_logvar[..., self.num_rot_components:],
            "entropy": entropy,
        }
        
        return output

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals:
            raise ValueError("FeatureField does not support computing normals")
        return self.get_outputs(ray_samples)
   