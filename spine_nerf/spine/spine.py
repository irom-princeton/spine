from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Optional
import os
import numpy as np
# import open_clip
import torch
import torchvision
from functools import cached_property
from PIL import Image
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer, NormalsRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.server.viewer_elements import *
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from nerfstudio.utils.misc import torch_compile

# from nerfstudio.models.base_model import get_rgba_image
from torch.nn import Parameter
import torch.nn.functional as F
import cv2
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)

from spine.encoders.image_encoder import BaseImageEncoder
from spine.spine_field import SPINEField
from spine.spine_fieldheadnames import SPINEFieldHeadNames
from spine.spine_renderers import MeanRenderer
from spine.data.utils.utils import apply_pca_colormap_return_proj
from spine.viewer_utils import ViewerUtils
from spine.spine_utils import (
    _lie_algebra_to_rotmat,
    _rotmat_to_lie_algebra,
    kl_divergence_loss_fn,
)



@dataclass
class SPINEModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: SPINEModel)
    num_spine_samples: int = 24
    "number of samples along the ray"
    output_semantics_during_training: bool = False
    """If True, output semantic-scene information during training. Otherwise, only output semantic-scene information during evaluation."""
    semantics_img_loss_weight: float = 1e-3
    cam_pose_autoencoder_loss_weight_trans: float = 1e-3
    """weight for the camera-pose-autoencoder-related term in the loss function."""
    cam_pose_autoencoder_loss_weight_rot: float = 1e-3
    """weight for the camera-pose-autoencoder-related term in the loss function."""
    kl_divergence_loss_weight: float = 1e-4
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
    
    # semantics computation override
    override_compute_semantics_no_compute: bool = False
    
    # Feature Field Positional Encoding
    feat_use_pe: bool = True
    feat_pe_n_freq: int = 6
    # Feature Field Hash Grid
    feat_num_levels: int = 12
    feat_log2_hashmap_size: int = 19
    feat_start_res: int = 16
    feat_max_res: int = 128
    feat_features_per_level: int = 8
    # Feature Field MLP Head
    feat_hidden_dim: int = 64
    feat_num_layers: int = 2
    
class SPINEModel(NerfactoModel):
    config: SPINEModelConfig

    def populate_modules(self):
        super().populate_modules()
        
        # parameters
        self.dino_image_encoder: BaseImageEncoder = self.kwargs["dino_image_encoder"]
        self.vggt_image_encoder: BaseImageEncoder = self.kwargs["vggt_image_encoder"]
        self.clip_image_encoder: BaseImageEncoder = self.kwargs["clip_image_encoder"]
        
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
        
        # renderers
        # self.renderer_normals = NormalsRenderer()
        # feature_dim = self.kwargs["metadata"]["feature_dim"]
        self.renderer_mean = MeanRenderer()

        # all fields
        self.spine_field = SPINEField(
            spatial_distortion=self.field.spatial_distortion,
            use_pe=self.config.feat_use_pe,
            pe_n_freq=self.config.feat_pe_n_freq,
            num_levels=self.config.feat_num_levels,
            log2_hashmap_size=self.config.feat_log2_hashmap_size,
            start_res=self.config.feat_start_res,
            max_res=self.config.feat_max_res,
            features_per_level=self.config.feat_features_per_level,
            hidden_dim=self.config.feat_hidden_dim,
            num_layers=self.config.feat_num_layers,
            base_semantic_field_output_dim=self.base_semantic_field_output_dim,
            img_semantic_embeds_input_dim=self.img_semantic_embeds_input_dim,
            lang_semantic_embeds_input_dim=self.lang_semantic_embeds_input_dim,
            cam_semantic_embeds_input_dim=self.cam_semantic_embeds_input_dim,
            img_semantic_feature_distillation_enabled=self.img_semantic_feature_distillation_enabled,
            lang_semantic_feature_distillation_enabled=self.lang_semantic_feature_distillation_enabled,
            camera_pose_estimation_enabled=self.camera_pose_estimation_enabled,
            learn_image_to_cam_embedding_pose_encoder=self.config.learn_image_to_cam_embedding_pose_encoder,
            dim_est_cam_pose=self.dim_est_cam_pose,
            dim_cam_pose_distribution_params=self.dim_cam_pose_distribution_params,
            dim_cam_pose_latent_distribution=self.dim_cam_pose_latent_distribution,
            cam_pose_output_dim=cam_pose_output_dim,
            num_gmm_components_cam_pose=self.config.num_gmm_components_cam_pose,
            num_trans_components=self.config.num_trans_components,
            num_rot_components=self.config.num_rot_components,
        )
        
        # Viewer Utils
        self.viewer_utils = ViewerUtils(image_encoder=self.clip_image_encoder)
        
        self.setup_gui()

    def setup_gui(self):
        self.viewer_utils.device = "cuda:0"
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: self.viewer_utils.reset_pca_proj())

        # Only setup GUI for language features if we're using CLIP
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(elem.value, is_positive=True),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(elem.value, is_positive=False),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=self.viewer_utils.softmax_temp,
            cb_hook=lambda elem: self.viewer_utils.update_softmax_temp(elem.value),
        )
        
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

    def get_outputs(
        self, 
        ray_bundle: RayBundle,
        compute_semantics: Optional[bool] = True,
    ):        
        """Takes in a ray bundlr and returns a dictionary of outputs.

        Args:
            ray_bundle: The Ray Bundle for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples, ray_bundle)
        spine_weights, best_ids = torch.topk(weights, self.config.num_spine_samples, dim=-2, sorted=False)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        spine_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)


        weights_list.append(weights)
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i],
                                                             ray_samples=ray_samples_list[i])
        
        # get semantic field outputs
        if compute_semantics and \
        (
            self.img_semantic_feature_distillation_enabled or 
            self.lang_semantic_feature_distillation_enabled
        ):
            # get the field outputs
            spine_field_outputs = self.spine_field.get_outputs(spine_samples)

            # image-space semantics
            outputs["img_semantics"] = None

            if self.img_semantic_feature_distillation_enabled:
                # image-space semantics
                outputs["img_semantics"] = self.renderer_mean(
                    embeds=spine_field_outputs[SPINEFieldHeadNames.IMG_SEMANTICS],
                    weights=spine_weights.detach()
                )
                
            # vision-language semantics
            outputs["lang_semantics"] = None
            
            if self.lang_semantic_feature_distillation_enabled:
                # vision-language semantics
                outputs["lang_semantics"] = self.renderer_mean(
                    embeds=spine_field_outputs[SPINEFieldHeadNames.LANG_SEMANTICS],
                    weights=spine_weights.detach()
                )
                
            
            # TODO: Needs a speed boost
            # compute the semantic PCA
            if not self.training and outputs["img_semantics"] is not None:
                # semantic PCA
                semantic_im_pca, _ = self.compute_semantic_pca(outputs["img_semantics"].detach())

                if len(outputs["rgb"].shape) >= 3:
                    # semantic PCA (image)
                    semantic_im_pca = semantic_im_pca.view(
                        *outputs["rgb"].shape[:2], 3
                    ).float()
                
                # semantic PCA (image)
                outputs["semantic_im_pca"] = semantic_im_pca
                
            if (
                self.config.output_semantics_during_training or not self.training
            ) and compute_semantics:
                if not self.config.override_compute_semantics_no_compute:
                    # Compute semantic inputs, e.g., composited similarity.
                    outputs = self.get_semantic_outputs(outputs=outputs)

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, 
        camera_ray_bundle: RayBundle,
        compute_semantics: Optional[bool] = True,
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        SPINE overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
            compute_semantics: option to compute semantics
        """
        # TODO: enable optional computation of semantics
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        
        return outputs
    
    def _get_outputs_nerfacto(self, ray_samples: RaySamples, ray_bundle: RayBundle):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        # compute RGB, depth, and accumulation
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
            
        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # loss dict
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
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
            # loss: semantic embeddings
            semantic_img_loss += self.config.semantics_img_loss_weight * (
                torch.nn.functional.mse_loss(
                    pred_semantics, 
                    batch_semantics,
                ) 
                + 
                (1 - torch.nn.functional.cosine_similarity(
                    pred_semantics, 
                    batch_semantics,
                    dim=-1,
                    )
                    ).mean()
            )
            
        if self.img_semantic_feature_distillation_enabled or self.lang_semantic_feature_distillation_enabled:
            # insert the loss component
            loss_dict["semantic_img_loss"]= semantic_img_loss
        
        # autoencoder loss for radiance field inversion
        cam_pose_autoencoder_loss = 0.0
            
        if self.camera_pose_estimation_enabled:
            # ground-truth latent embeddings
            gt_cam_latents = batch["img_semantics"][0].half()[None]
            gt_cam_latents = F.normalize(gt_cam_latents, p=2, dim=-1)
            
            # encode the camera pose
            cam_pose_enc, cam_trans, cam_rot = self.spine_field.encode_camera_pose(batch["camera_to_world"])
            
            # decode the camera latent embeddings
            cam_pose_dec = None
            
            # TODO: Not yet supported
            if self.config.learn_image_to_cam_embedding_pose_encoder:
                cam_pose_dec = self.spine_field.decode_camera_pose(cam_pose_enc)
             
            # end-to-end supervision or teacher forcing
            cam_pose_dec_output = self.spine_field.decode_camera_pose(
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
                
            # insert the loss component
            loss_dict["cam_pose_autoencoder_loss"]= cam_pose_autoencoder_loss
           
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["spine"] = list(self.spine_field.parameters())
        return param_groups
