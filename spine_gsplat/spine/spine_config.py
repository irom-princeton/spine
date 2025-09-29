"""
SPINE configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
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


spine_method = MethodSpecification(
    config=TrainerConfig(
        method_name="spine",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        # gradient_accumulation_steps={"camera_opt": 100},
        pipeline=SPINEPipelineConfig(
            datamanager=SPINEDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=SPINEModelConfig(),
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
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "base_semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
            },
            "img_semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
            },
            "lang_semantic_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
            },
            "cam_pose_to_embedding": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
            },
            "embedding_to_cam_pose": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-5),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for SPINE",
)
