from dataclasses import dataclass, field
from typing import Tuple, Type, List

import torch
import torchvision

import gc

from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from torchvision import transforms
from tqdm import tqdm

from spine.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from spine.encoders.models.dino import DinoV2Encoder


@dataclass
class DINONetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: DINONetwork)
    model_name: str =  "dinov3_vits16"  # "dinov2_vits14", "dinov3_vits14"
    feature_key_patch: str = "x_norm_patchtokens"
    feature_key_cls: str = "x_norm_clstoken"
    feature_key: str = feature_key_patch
    embed_dims: int = 384
    batch_size: int = 1
    feature_img_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
        }

class DINONetwork(BaseImageEncoder):
    def __init__(self, config: DINONetworkConfig):
        super().__init__()
        self.config = config
        # self.im_h = None
        # self.im_w = None
        self.device = self.config.device
        self.feature_img_size = self.config.feature_img_size
        self.model = None
        
    def _load_model(self, feature_key: str = None):
        """
        Load the model
        """
        # feature key
        if feature_key is None:
            feature_key = self.config.feature_key_patch
            
        self.model = DinoV2Encoder(
            name=self.config.model_name,
            feature_key=feature_key,
        )
        self.model.eval()
        self.model.to("cuda")  
          
    def _del_model(self):
        # Delete and clear memory to be safe
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
        # update the reference to the model
        self.model = None
        
        # self.tokenizer = clip.tokenize
        # self.clip_n_dims = self.config.clip_n_dims

        # self.positives = ["hand sanitizer"]
        # self.negatives = self.config.negatives
        # with torch.no_grad():
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
        #     self.pos_embeds = self.model.encode_text(tok_phrases)
        #     tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
        #     self.neg_embeds = self.model.encode_text(tok_phrases)
        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        # self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # assert (
        #     self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        # ), "Positive and negative embeddings must have the same dimensionality"
        # assert (
        #     self.pos_embeds.shape[1] == self.clip_n_dims
        # ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        # examine for the model name
        if "dinov3" in self.config.model_name.lower():
            self.base_model_name = "DINOv3"
        else:
            self.base_model_name = "DINOv2"
        
        return "{}_{}".format(self.base_model_name, self.config.model_name)

    @property
    def embedding_dim(self) -> int:
        return self.config.embed_dims

    @staticmethod
    def default_img_transform(img_size=224):
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                # transforms.CenterCrop(img_size),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        
    def set_positives(self, text_list):
        pass

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        pass
    
    def encode_image(
        self, 
        image_list, 
        img_size=None, 
        return_camera_feats: bool = True,
        return_img_feats: bool = True,
    ):
        # # number of patches
        # num_patches = (torch.tensor(image_list.shape[-2:]) / self.model.patch_size).int()
        
        # if num_patches[0] != num_patches[0].int() or num_patches[1] != num_patches[1].int():
        #     # image size for resize
        #     out_img_size = (torch.tensor(image_list.shape[-2:]) * torch.floor(num_patches)).int().cpu().numpy().tolist()
        # else:
        #     out_img_size = torch.tensor(image_list.shape[-2:]).int().cpu().numpy().tolist()
        
        # load the model, if necessary
        self._load_model()
        
        if img_size is None:
            img_size = self.feature_img_size
            
        # initialize image transform
        self.img_transform = self.default_img_transform(img_size=img_size)
        
        # apply the image transforms
        image_list = [
            self.img_transform(img)
            for img in image_list
        ]
        image_list = torch.stack(image_list).to(self.device)
        
        # ----------------------------------------------------- #
        # Compute Camera Embeddings (CLS token)
        # ----------------------------------------------------- #
        
        # camera embeddings
        self.cam_embeds = None
            
        if return_camera_feats:
            # delete the model
            self._del_model()
            
            # load the model
            self._load_model(feature_key=self.config.feature_key_cls)
            
            with torch.no_grad():
                # compute the camera embeddings
                self.cam_embeds = []
            
                for img in tqdm(image_list, desc="Processing images for camera embeddings"):
                    # computes the camera embeddings independently
                    cam_embeds = self.model(img.unsqueeze(dim=0))
                
                    self.cam_embeds.append(cam_embeds.cpu())
                     
                # concatenate the embeddings
                if len(self.cam_embeds) != 0:
                    self.cam_embeds = torch.cat(self.cam_embeds, dim=0).squeeze(1)
            
        # ----------------------------------------------------- #
        # Compute Image Embeddings (Patch Tokens)
        # ----------------------------------------------------- #
        
        # image embeddings
        self.img_embeds = None
        
        if return_img_feats:
            # delete the model
            self._del_model()
            
            # load the model
            self._load_model(feature_key=self.config.feature_key_patch)

            with torch.no_grad():
                # compute the image embeddings
                self.img_embeds = []
            
                for img in tqdm(image_list, desc="Processing images"):
                    # computes the camera embeddings independently
                    img_embeds = self.model(img.unsqueeze(dim=0))
                
                    self.img_embeds.append(img_embeds.cpu())
                        
                # concatenate the embeddings
                if len(self.img_embeds) != 0:
                    self.img_embeds = torch.cat(self.img_embeds, dim=0)
            
            # number of patches
            num_patches = (torch.tensor(image_list.shape[-2:]) / self.model.patch_size).int()

            self.img_embeds = rearrange(self.img_embeds, "b (h w) c -> b h w c", h=num_patches[0], w=num_patches[1])
            
            # print status
            print(F"Finished extracting DINOv2 features for {len(image_list)} images with shape {self.img_embeds.shape}.")

        # augmented image embeddings
        self.aug_embeds = None
        
        # Delete and clear memory to be safe
        self._del_model()

        return self.cam_embeds, self.img_embeds, self.aug_embeds
