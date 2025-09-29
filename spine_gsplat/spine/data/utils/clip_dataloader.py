
import json
import os
from pathlib import Path

import numpy as np
import torch
from spine.data.utils.feature_dataloader import FeatureDataloader
from spine.encoders.clip_encoder import CLIPNetwork
import torchvision.transforms.functional as TF
from tqdm import tqdm
from spine.data.utils.utils import apply_pca_colormap_return_proj

import gc


class CLIPDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: CLIPNetwork,
        image_list: torch.Tensor = None,
        cache_path: str = None,
        return_camera_feats: bool = True,
        return_img_feats: bool = True,
    ):
        self.model = model
        self.data_dict = {}
        self.image_list = image_list
        self.return_camera_feats = return_camera_feats
        self.return_img_feats = return_img_feats

        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, image_idx):
        output = []
        if self.cam_data is not None:
            output.append(
                self.cam_data[image_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
   
        if self.img_data is not None:
            output.append(
                self.img_data[image_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
            
        if self.data_aug_embeds is not None:
            output.append(
                self.data_aug_embeds[image_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
            
        return tuple(output)

    def load(self):
        super().load()
  
        # Optionally, add scaling/interpolation logic if needed
        # self._init_scaling_factors()

    def create(self, image_list):
        # Similar to DINODataloader, expects encode_image to return cam_data, img_data, data_aug_embeds
        # self.img_data = self.model.encode_image(image_list).cpu().detach()
        
        # compute embeddings
        self.cam_data, self.img_data, self.data_aug_embeds = self.model.encode_image(
            image_list,
            return_camera_feats=self.return_camera_feats,
            return_img_feats=self.return_img_feats,
        )
        
        # move to CPU
        if self.cam_data is not None:
            self.cam_data = self.cam_data.cpu().detach()
            
        if self.img_data is not None:
            self.img_data = self.img_data.cpu().detach()
            
        if self.data_aug_embeds is not None:
            self.data_aug_embeds = self.data_aug_embeds.cpu().detach()
            
            
    def _init_scaling_factors(self):
        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = self.img_data.shape[1:3]
        assert len(self.model.im_h) == 1, "All images must have the same height"
        assert len(self.model.im_w) == 1, "All images must have the same width"
        self.model.im_h, self.model.im_w = self.model.im_h.pop(), self.model.im_w.pop()
        self.model.scale_h = feat_h / self.model.im_h
        self.model.scale_w = feat_w / self.model.im_w
         
    def interpolate_data(self, data, output_size) -> None:
        return torch.nn.functional.interpolate(
            input=data,
            size=output_size,
            mode="nearest",
        )
