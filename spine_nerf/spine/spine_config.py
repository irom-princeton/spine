"""
SPINE configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from spine.data.spine_datamanager import SPINEDataManagerConfig
from spine.spine import SPINEModelConfig
from spine.spine_pipeline import SPINEPipelineConfig

from spine.encoders.dino_encoder import DINONetworkConfig
from spine.encoders.vggt_encoder import VGGTNetworkConfig
from spine.encoders.clip_encoder import CLIPNetworkConfig

"""
Swap out the network config to use OpenCLIP or CLIP here.
"""
# from spine.encoders.clip_sam_encoder import CLIPSAMNetworkConfig


spine_method = MethodSpecification(
    config=TrainerConfig(
        method_name="spine",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SPINEPipelineConfig(
            datamanager=SPINEDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99), # scene_scale=5.494825839996338
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=SPINEModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_spine_samples=24,
            ),
            #  You can swap the type of input encoder by specifying different clip_model_types, e.g., "RN50x64," "ViT-B/16."
            dino_network=DINONetworkConfig(
                model_name=DINONetworkConfig().model_name,
                feature_key=DINONetworkConfig().feature_key_patch,
            ),
            vggt_network=VGGTNetworkConfig(
                model_name=VGGTNetworkConfig().model_name,
                feature_key=VGGTNetworkConfig().feature_key,
            ),
            clip_network=CLIPNetworkConfig(
                model_name=CLIPNetworkConfig().model_name,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "spine": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for SPINE",
)
spine_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="spine-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SPINEPipelineConfig(
            datamanager=SPINEDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=SPINEModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_spine_samples=32,
            ),
            #  You can swap the type of input encoder by specifying different clip_model_types, e.g., "RN50x64," "ViT-B/16."
            dino_network=DINONetworkConfig(
                model_name=DINONetworkConfig().model_name,
                feature_key=DINONetworkConfig().feature_key_patch,
            ),
            vggt_network=VGGTNetworkConfig(
                model_name=VGGTNetworkConfig().model_name,
                feature_key=VGGTNetworkConfig().feature_key,
            ),
            clip_network=CLIPNetworkConfig(
                model_name=CLIPNetworkConfig().model_name,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "spine": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=3000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A larger version of SPINE with a higher memory footprint, bigger CLIP model, and more hashgrid capacity",
)

spine_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="spine-lite",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=SPINEPipelineConfig(
            datamanager=SPINEDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=SPINEModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_spine_samples=12,
            ),
            #  You can swap the type of input encoder by specifying different clip_model_types, e.g., "RN50x64," "ViT-B/16."
            dino_network=DINONetworkConfig(
                model_name=DINONetworkConfig().model_name,
                feature_key=DINONetworkConfig().feature_key_patch,
            ),
            vggt_network=VGGTNetworkConfig(
                model_name=VGGTNetworkConfig().model_name,
                feature_key=VGGTNetworkConfig().feature_key,
            ),
            clip_network=CLIPNetworkConfig(
                model_name=CLIPNetworkConfig().model_name,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000),
            },
            "spine": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=7000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A lightweight version of SPINE designed to work on smaller GPUs",
)
