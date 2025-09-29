import json
import os
from pathlib import Path

import numpy as np
import torch
from spine.data.utils.feature_dataloader import FeatureDataloader
from spine.encoders.dino_encoder import DINONetwork
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
from spine.data.utils.utils import apply_pca_colormap_return_proj

import gc


class DINODataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: DINONetwork,
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
        # image_idx: index of the image in the training dataset
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
        # load the embeddings
        super().load()

        # # Determine scaling factors for nearest neighbor interpolation
        # feat_h, feat_w = self.data.shape[1:3]
        # assert len(self.model.im_h) == 1, "All images must have the same height"
        # assert len(self.model.im_w) == 1, "All images must have the same width"
        # self.model.im_h, self.model.im_w = self.model.im_h.pop(), self.model.im_w.pop()
        # self.model.scale_h = feat_h / self.model.im_h
        # self.model.scale_w = feat_w / self.model.im_w

    def create(self, image_list):
        # self.img_data = []
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
        
        # # OOM Error
        # # interpolate the data
        # self.data = self.interpolate_data(
        #     data=self.data,
        #     output_size=(len(image_list), self.data[1], *self.image_list[0].shape[-2:]), # B x C x H x W
        # )
    
    def interpolate_data(self, data, output_size) -> None:
        # interpolate the image
        return torch.nn.functional.interpolate(
            input=data,
            size=output_size,
            mode="nearest",
            
        )
               