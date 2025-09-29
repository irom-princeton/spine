from typing import Literal
import torch
import torch.nn as nn
from vggt.models.vggt import VGGT
import numpy as np

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class VGGTEncoder(nn.Module):
    def __init__(self, name, feature_key: Literal["point_head", "depth_head"] = "point_head"):
        super().__init__()
        self.name = name
        self.base_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        # self.emb_dim = self.base_model.num_features
        self.feature_key = feature_key.lower()
        # if feature_key == "x_norm_patchtokens":
        #     self.latent_ndim = 2
        # elif feature_key == "x_norm_clstoken":
        #     self.latent_ndim = 1
        # else:
        #     raise ValueError(f"Invalid feature key: {feature_key}")
        
        # self.patch_size = self.base_model.patch_size

    def forward(
        self, 
        images,
        return_camera_feats: bool = True,  # always True by design
        return_img_feats: bool = True,
        augment_features: bool = False, # augment point/depth-head features with intermediate-layer features
        intermediate_layer_idx: int = 23, #11,
    ):
        if len(images.shape) == 4:
            images = images[None]  # add batch dimension
            
        # compute tokens
        aggregated_tokens_list, ps_idx = self.base_model.aggregator(images)

        # return the features for camera pose prediction (the last tensor and the first token in this tensor)
        cam_emb = aggregated_tokens_list[-1][:, :, 0]

        # no image features
        img_emb = None
        
        # no augmented features
        aug_emb = None
        
        if return_img_feats:
            # return the image features
            # get the features for dense prediction tasks
            if self.feature_key == "point_head":
                self.base_model.point_head.feature_only = True # required for DPT features only
                img_emb = self.base_model.point_head(aggregated_tokens_list, images, ps_idx)
            elif self.feature_key == "depth_head":
                self.base_model.depth_head.feature_only = True # required for DPT features only
                img_emb = self.base_model.depth_head(aggregated_tokens_list, images, ps_idx)
            else:
                raise ValueError(f"Unsupported feature type: {self.feature_key} provided to VGGT! Supported types are ('point_head', 'depth_head').")
                
            if augment_features:
                aug_emb = aggregated_tokens_list[intermediate_layer_idx][:, :, ps_idx:]
            
        return cam_emb, img_emb, aug_emb

if __name__ == "__main__":
    # test
    model = VGGTEncoder(name="vggt", feature_key="point_head").to(device)
    breakpoint()