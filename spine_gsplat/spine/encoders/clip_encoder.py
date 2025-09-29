from dataclasses import dataclass, field
from typing import Tuple, Type, List

import torch
import torchvision

import gc

from einops import rearrange
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from spine.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from spine.encoders.models.clip import CLIPEncoder

@dataclass
class CLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: CLIPNetwork)
    model_name: str = "RN50x64"  # "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    embed_dims: int = 512
    batch_size: int = 1
    feature_img_size = 224
    skip_center_crop: bool = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def id_dict(cls):
        return {
            "model_name": cls.model_name,
            "pretrained": cls.pretrained,
            "skip_center_crop": cls.skip_center_crop, 
        }


class CLIPNetwork(BaseImageEncoder):
    def __init__(self, config: CLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.device = self.config.device
        self.feature_img_size = self.config.feature_img_size
        # self.im_h = None
        # self.im_w = None
        self.model = None
        
    def _load_model(self):
        """
        Load the model
        """
        if self.model is None:
            self.model = CLIPEncoder(
                name=self.config.model_name,
                skip_center_crop=True, #TODO: Make this configurable
            )
            self.model.eval()
            self.model.to(self.device)

    def _del_model(self):
        # Delete and clear memory to be safe
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
        # update the reference to the model
        self.model = None
        
    @property
    def name(self) -> str:
        return f"CLIP_{self.config.model_name}"

    @property
    def embedding_dim(self) -> int:
        return self.config.embed_dims

    def encode_image(
        self, 
        image_list, 
        img_size=None, 
        return_camera_feats: bool = True,
        return_img_feats: bool = True,
    ):  # load the model, if necessary
        self._load_model()
        
        if img_size is None:
            img_size = self.feature_img_size

        # ----------------------------------------------------- #
        # Compute Image Embeddings (Patch Tokens)
        # ----------------------------------------------------- #
        
        # Get CLIP embeddings for the images
        self.img_embeds = []
        
        with torch.no_grad():
            for i in tqdm(
                range(0, len(image_list), 
                      CLIPNetworkConfig.batch_size),
                desc="Extracting CLIP features",
            ):
                # compute the image embeddings
                batch = image_list[i : i + CLIPNetworkConfig.batch_size]
                self.img_embeds.append(self.model(batch)) # shape: [b, 7, 7, 512]

        # concatenate the image embeddings
        self.img_embeds = torch.cat(self.img_embeds, dim=0)

        # print status
        print(F"Finished extracting CLIP features for {len(image_list)} images with shape {self.img_embeds.shape}.")

        # Determine scaling factors for nearest neighbor interpolation
        # feat_h, feat_w = self.img_embeds.shape[1:3]
        # im_h, im_w = image_list[0].shape[-2:]
        # self.scale_h = feat_h / im_h
        # self.scale_w = feat_w / im_w
        
        # Delete and clear memory to be safe
        self._del_model()
        
        # no camera and augmented image embeddings
        self.cam_embeds, self.aug_embeds = None, None

        return self.cam_embeds, self.img_embeds, self.aug_embeds
