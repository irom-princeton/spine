from dataclasses import dataclass, field
from typing import Tuple, Type, List

import torch
import torchvision
import torch.nn as nn

import gc
import sys
from pathlib import Path

from einops import rearrange
from torchvision.transforms import CenterCrop, Compose

from spine.clip import clip


class CLIPEncoder(nn.Module):
    def __init__(self,
                 name,
                 skip_center_crop: bool = True,
                 ):
        super().__init__()
        
        # model's name
        self.name = name
        
        # option to skip center crop
        self.skip_center_crop = skip_center_crop
        
        # load the model
        self.model, self.preprocess = clip.load(name)
        
        # move to GPU
        self.model.to("cuda")
        
        # Patch the preprocess if we want to skip center crop
        if self.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in self.preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            self.preprocess = Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, CenterCrop)]
            )
            print("Skipping center crop")
            
        # image transform
        self.rgb_image_transform = torchvision.transforms.ToPILImage()

        if hasattr(self.model.visual, "patch_size"):
            self.patch_size = self.model.visual.patch_size
        else:
            self.patch_size = None

    def forward(self, x):
        # Preprocess the images
        images = [self.rgb_image_transform(x[i,:,:,:]).convert("RGB") for i in range(x.size()[0])]
        preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        preprocessed_images = preprocessed_images.to("cuda")  # (b, 3, h, w)
        # print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")
        
        if self.name in clip._MODELS:
            # extract the CLIP embeddings
            embeddings = self.model.get_patch_encodings(preprocessed_images)
            
            # Reshape embeddings from flattened patches to patch height and width
            h_in, w_in = preprocessed_images.shape[-2:]
            if self.name.startswith("ViT"):
                h_out = h_in // self.model.visual.patch_size
                w_out = w_in // self.model.visual.patch_size
            elif self.name.startswith("RN"):
                h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
                w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
                h_out, w_out = int(h_out), int(w_out)
            else:
                raise ValueError(f"Unknown CLIP model name: {self.name}")
            
            embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
            # print(f"Extracted CLIP embeddings of shape {embeddings.shape}")
        else:
            # Get CLIP embeddings for the images
            embeddings = self.model.encode_image(preprocessed_images)
            # print(f"Extracted CLIP embeddings of shape {embeddings.shape}")
            assert (
                len(embeddings.shape) > 2
            ), f"The embeddings for each image should be two-dimensional, but has shape {embeddings.shape[1:]}."
            
        return embeddings