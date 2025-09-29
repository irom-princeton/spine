import json
import os
from pathlib import Path

import numpy as np
import torch
from spine.data.utils.feature_dataloader import FeatureDataloader
from spine.encoders.vggt_encoder import VGGTNetwork
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
from spine.data.utils.utils import apply_pca_colormap_return_proj

import gc


class VGGTDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: VGGTNetwork,
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

    def __call__(self, ray_indices):
        # img_points: (B, 3) # (img_ind, x, y)
        
        # image index from ray bundle
        image_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        
        # output
        output = []
        if self.cam_data is not None:
            output.append(
                self.cam_data[image_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
   
        if self.img_data is not None:
            output.append(
                self.img_data[image_idx, y_idx, x_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
            
        if self.data_aug_embeds is not None:
            output.append(
                self.data_aug_embeds[image_idx, y_idx, x_idx].type(torch.float32).to(self.device)
            )
        else:
            output.append(None)
            
        return tuple(output)
    
    def load(self):
        # load the embeddings
        super().load()
  
        # Optionally, add scaling/interpolation logic if needed
        self._init_scaling_factors()

    def create(self, image_list):
        # self.data = []
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
            
        # Optionally, add scaling/interpolation logic if needed
        self._init_scaling_factors()
            
    def _init_scaling_factors(self):
        # Determine scaling factors for nearest neighbor interpolation
        
        # TODO: Add a check to ensure that all images have the same dimensions
        
        # image dimension (width and height)
        im_h, im_w = self.image_list[0].shape[-2:]
        
        # embedding dimension
        feat_h, feat_w = self.img_data.shape[1:3]
        
        # scaling factors
        self.scale_h = feat_h / im_h
        self.scale_w = feat_w / im_w
        
    def interpolate_data(self, data, output_size) -> None:
        # interpolate the image
        return torch.nn.functional.interpolate(
            input=data,
            size=output_size,
            mode="nearest",
            
        )
               