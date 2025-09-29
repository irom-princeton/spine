# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Datamanager.
"""

from __future__ import annotations

import random
import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import numpy
import torch
# import torchvision
import yaml
from copy import deepcopy

from nerfstudio.cameras.cameras import Cameras, CameraType
from rich.progress import Console

CONSOLE = Console(width=120)
from spine.encoders.image_encoder import BaseImageEncoder
from spine.data.utils.dino_dataloader import DINODataloader
from spine.data.utils.vggt_dataloader import VGGTDataloader
from spine.data.utils.clip_dataloader import CLIPDataloader
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    ForwardRef,
    get_origin,
    get_args,
)

from PIL import Image
import pdb

@dataclass
class SPINEDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: SPINEDataManager)
    # image-space semantic feature extractor
    semantic_extractor: Literal["dino", "vggt"] = "vggt"
    # vision-language semantics model
    language_semantics_model: Literal["clip"] = "clip"
    # option to distill camera features
    distill_cam_feats: bool = True
    # option to distill image features
    distill_img_feats: bool = True
    # option to distill vision-language semantics
    distill_lang_semantics: bool = True
    

class SPINEDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: SPINEDataManagerConfig

    def __init__(
        self,
        config: SPINEDataManagerConfig,
        device: Union[torch.device, str] = "cuda:0",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        # image encoders
        self.dino_image_encoder: BaseImageEncoder = kwargs["dino_image_encoder"]
        self.vggt_image_encoder: BaseImageEncoder = kwargs["vggt_image_encoder"]
        self.clip_image_encoder: BaseImageEncoder = kwargs["clip_image_encoder"]

        # images
        images = [self.train_dataset[i]["image"][..., :3].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)
    
        # path to SPINE
        parent_path = Path(__file__).parent.parent.parent.parent.resolve()

        # output directory
        cache_dir = f"{parent_path}/outputs/gsplat/{self.config.dataparser.data.name}/dataloader"
        
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        if "dino" in self.config.semantic_extractor.lower():
            # image encoder 
            image_encoder = self.dino_image_encoder
            
            # dataloader
            feat_dataloader = DINODataloader
            
        elif "vggt" in self.config.semantic_extractor.lower():
            # image encoder 
            image_encoder = self.vggt_image_encoder
            
            # dataloader
            feat_dataloader = VGGTDataloader
        else:
            raise RuntimeError(f"Unsupported image-space semantic model specified: {self.config.semantic_extractor}!")
        
        # cache directory for image-space semantics features
        semantics_cache_path = Path(osp.join(cache_dir, f"{image_encoder.name}.npy"))
    
        # extract image-space features
        self.img_semantics_interpolator = feat_dataloader(
            image_list=images,
            device=self.device,
            cfg={
                "image_shape": list(images.shape[2:4]),
                "model_name": image_encoder.name,
            },
            cache_path=semantics_cache_path,
            model=image_encoder,
            return_camera_feats=self.config.distill_cam_feats,
            return_img_feats=self.config.distill_img_feats,
        )
        
        if "clip" in self.config.language_semantics_model.lower():
            # image encoder 
            image_encoder = self.clip_image_encoder
            
            # dataloader
            feat_dataloader = CLIPDataloader
            
        # vision-language semantics cache path
        lang_semantics_cache_path = Path(osp.join(cache_dir, f"{image_encoder.name}.npy"))
            
        # extract image-space features
        self.lang_semantics_interpolator = feat_dataloader(
            image_list=images,
            device=self.device,
            cfg={
                "image_shape": list(images.shape[2:4]),
                "model_name": image_encoder.name,
            },
            cache_path=lang_semantics_cache_path,
            model=image_encoder,
            return_camera_feats=self.config.distill_cam_feats,
            return_img_feats=self.config.distill_img_feats,
        )
        # free memory
        torch.cuda.empty_cache()

        # feature dimension
        feat_dim = {}
        
        # image-space semantics
        if self.img_semantics_interpolator.cam_data is not None:
            feat_dim["cam_feats"] = self.img_semantics_interpolator.cam_data.shape[-1]
            
        if self.img_semantics_interpolator.img_data is not None:
            feat_dim["img_feats"] = self.img_semantics_interpolator.img_data.shape[-1]
        
        if self.img_semantics_interpolator.data_aug_embeds is not None:
            feat_dim["aug_feats"] = self.img_semantics_interpolator.data_aug_embeds.shape[-1]
            
        # vision-language semantics
        if self.lang_semantics_interpolator.img_data is not None:
            feat_dim["lang_img_feats"] = self.lang_semantics_interpolator.img_data.shape[-1]
            
        # feature dimensions
        self.train_dataset.metadata["feature_dim"] = feat_dim
            
        # camera pose estimation and semantic image feature distillation options
        self.train_dataset.metadata["camera_pose_estimation_enabled"] = self.config.distill_cam_feats
        self.train_dataset.metadata["img_semantic_feature_distillation_enabled"] = self.config.distill_img_feats
        self.train_dataset.metadata["lang_semantic_feature_distillation_enabled"] = self.config.distill_lang_semantics


    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        # TODO: Add functionality for RGBA images
        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)
        
        # # DINO/VGGT/CLIP embeddings
        # image-space semantics
        data["img_semantics"] = self.img_semantics_interpolator(image_idx)# .to(self.device)
        
        # vision-language semantics
        data["lang_semantics"] = self.lang_semantics_interpolator(image_idx)
        
        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
            
        camera.metadata["cam_idx"] = image_idx
        
        return camera, data
