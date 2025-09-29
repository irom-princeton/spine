from dataclasses import dataclass, field
from typing import Tuple, Type, List

import torch
import torchvision

import gc

from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from torchvision import transforms as TF
from torchvision import transforms
from tqdm import tqdm
import warnings

from spine.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from spine.encoders.models.vggt import VGGTEncoder


from vggt.utils.load_fn import load_and_preprocess_images


@dataclass
class VGGTNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: VGGTNetwork)
    model_name: str =  "vggt" # which uses "dinov2_vitl14_reg"
    # dino_base_model: str = "dinov2_vitl14_reg"
    feature_key: str = "point_head"  # depth_head, point_head
    embed_dims: int = 128
    batch_size: int = 1
    feature_img_size = (336, 518)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
        }

class VGGTNetwork(BaseImageEncoder):
    def __init__(self, config: VGGTNetworkConfig):
        super().__init__()
        self.config = config
        # self.im_h = None
        # self.im_w = None
        self.device = self.config.device
        self.feature_img_size = self.config.feature_img_size
        self.model = None
        
    def _load_model(self):
        """
        Load the model
        """
        self.model = VGGTEncoder(
            name=self.config.model_name,
            feature_key=self.config.feature_key
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
        
    @property
    def name(self) -> str:
        return "VGGT_{}".format(self.config.model_name)

    @property
    def embedding_dim(self) -> int:
        return self.config.embed_dims

    # adapted from VGGT
    @staticmethod
    def preprocess_images(image_list, mode="crop"):
        """
        A quick start function to load and preprocess images for model input.
        This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

        Args:
            image_list (list): List of images
            mode (str, optional): Preprocessing mode, either "crop" or "pad".
                                - "crop" (default): Sets width to 518px and center crops height if needed.
                                - "pad": Preserves all pixels by making the largest dimension 518px
                                and padding the smaller dimension to reach a square shape.

        Returns:
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

        Raises:
            ValueError: If the input list is empty or if mode is invalid

        Notes:
            - Images with different dimensions will be padded with white (value=1.0)
            - A warning is printed when images have different shapes
            - When mode="crop": The function ensures width=518px while maintaining aspect ratio
            and height is center-cropped if larger than 518px
            - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
            and the smaller dimension is padded to reach a square shape (518x518)
            - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        """
        # Check for empty list
        if len(image_list) == 0:
            raise ValueError("At least 1 image is required")

        # Validate mode
        if mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        images = []
        shapes = set()
        to_tensor = TF.ToTensor()
        target_size = 518
    
        # to PIL Image    
        rgb_image_transform = torchvision.transforms.ToPILImage()

        # First process all images and collect their shapes
        for img in image_list:
            # convert to PIL Image
            img = rgb_image_transform(img)
            
            # If there's an alpha channel, blend onto white background:
            if img.mode == "RGBA":
                # Create white background
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                # Alpha composite onto the white background
                img = Image.alpha_composite(background, img)

            # Now convert to "RGB" (this step assigns white for transparent areas)
            img = img.convert("RGB")

            width, height = img.size

            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((img.shape[1], img.shape[2]))
            images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes
        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in images:
                h_padding = max_height - img.shape[1]
                w_padding = max_width - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
                padded_images.append(img)
            images = padded_images

        images = torch.stack(images)  # concatenate images

        # Ensure correct shape when single image
        if len(image_list) == 1:
            # Verify shape is (1, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)

        return images
        
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
        # load the model, if necessary
        self._load_model()
        
        # data type
        spec_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=spec_dtype):
                # preprocess the images
                images = self.preprocess_images(image_list).to(self.device)
                
                # compute the image 
                # self.embeddings, self.aug_embeds = self.model(images, return_camera_features=return_camera_features_only)
                
                self.cam_embeds, self.img_embeds, self.aug_embeds = [], [], []
            
                for img in tqdm(images, desc="Processing images"):
                    # computes the camera embeddings independently
                    cam_embeds, img_embeds, aug_embeds = self.model(
                        img.unsqueeze(dim=0), 
                        return_camera_feats=return_camera_feats,
                        return_img_feats=return_img_feats,
                    )
                
                    if cam_embeds is not None:
                        # append embeddings
                        self.cam_embeds.append(cam_embeds.cpu())
                        
                    if img_embeds is not None:
                        # append embeddings
                        self.img_embeds.append(img_embeds.cpu())
                    
                    if aug_embeds is not None:
                        # append embeddings
                        self.aug_embeds.append(aug_embeds.cpu())
                        
                # concatenate the embeddings
                if len(self.cam_embeds) != 0:
                    self.cam_embeds = torch.cat(self.cam_embeds, dim=1)
                else:
                    self.cam_embeds = None
                    
                if len(self.img_embeds) != 0:
                    self.img_embeds = torch.cat(self.img_embeds, dim=1)
                else:
                    self.img_embeds = None
                    
                if len(self.aug_embeds) != 0:
                    self.aug_embeds = torch.cat(self.aug_embeds, dim=1)
                else:
                    self.aug_embeds = None
            
                # remove the batch dimension
                if self.cam_embeds is not None:
                    self.cam_embeds = self.cam_embeds.squeeze()
                if self.img_embeds is not None:
                    self.img_embeds = self.img_embeds.squeeze().moveaxis(1, -1)  # (N, C, H, W) -> (N, H, W, C)
                
                if self.aug_embeds is not None:
                    self.aug_embeds = self.aug_embeds.squeeze()
                    
                    warnings.warn("Assuming a patch embed size of 14! Please update if necessary!")
                    self.aug_embeds = self.aug_embeds.reshape(
                        self.aug_embeds.shape[0],
                        images.shape[-2] // 14,
                        images.shape[-1] // 14,
                        -1,
                    )
                
        # print status
        print(
            f"Finished extracting VGGT features for {len(image_list)} images with cam embeds shape {self.cam_embeds.shape}, \
            img embeds shape {self.img_embeds.shape if self.img_embeds is not None else None} \
            and {self.aug_embeds.shape if self.aug_embeds is not None else None}."
        )

        # Delete and clear memory to be safe
        self._del_model()
        
        return self.cam_embeds, self.img_embeds, self.aug_embeds
