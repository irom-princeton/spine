import json
import os
import typing
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch


class FeatureDataloader(ABC):
    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.Tensor, # (N, 3, H, W)
            cache_path: Path,
    ):
        self.cfg = cfg
        self.device = device
        self.cache_path = cache_path
        self.cam_data = None # only expect data to be cached, nothing else
        self.img_data = None # only expect data to be cached, nothing else
        self.data_aug_embeds = None # only expect data to be cached, nothing else
        self.try_load(image_list) # don't save image_list, avoid duplicates

    @abstractmethod
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        pass

    @abstractmethod
    def create(self, image_list: torch.Tensor):
        pass

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        
        # load any cached data files
        # filepath and extension
        file_path_with_ext = os.path.splitext(self.cache_path)
        file_path_without_ext, file_ext = file_path_with_ext[0], file_path_with_ext[1]
        
        # print
        print(f"Loading camera/image features from {file_path_without_ext}...")
        
        # camera data
        cam_data_file_path = Path(f"{file_path_without_ext}_cam_data{file_ext}")
        
        if cam_data_file_path.exists():
            # To prevent out-of-memory errors (move to CPU)
            self.cam_data = torch.from_numpy(np.load(cam_data_file_path)).float()
            
        # image data
        img_data_file_path = Path(f"{file_path_without_ext}_img_data{file_ext}")
        
        if img_data_file_path.exists():
            # To prevent out-of-memory errors (move to CPU)
            self.img_data = torch.from_numpy(np.load(img_data_file_path)).float()
        
        # augmented data (embeds)
        aug_data_file_path = Path(f"{file_path_without_ext}_aug_data{file_ext}")
        
        if aug_data_file_path.exists():
            # To prevent out-of-memory errors (move to CPU)
            self.data_aug_embeds = torch.from_numpy(np.load(aug_data_file_path)).float()
            
    def save(self):
        os.makedirs(self.cache_path.parent, exist_ok=True)
        cache_info_path = self.cache_path.with_suffix(".info")

        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
            
        # filepath and extension
        file_path_with_ext = os.path.splitext(self.cache_path)
        file_path_without_ext, file_ext = file_path_with_ext[0], file_path_with_ext[1]
        
        # print
        print(f"Saving camera/image features to {file_path_without_ext}...")
        
        # camera data
        if self.cam_data is not None:
            np.save(Path(f"{file_path_without_ext}_cam_data{file_ext}"), self.cam_data)
            
        # image data
        if self.img_data is not None:
            np.save(Path(f"{file_path_without_ext}_img_data{file_ext}"), self.img_data)
        
        # augmented data (embeds)
        if self.data_aug_embeds is not None:
            np.save(Path(f"{file_path_without_ext}_aug_data{file_ext}"), self.data_aug_embeds)

    def try_load(self, img_list: torch.Tensor):
        try:
            self.load()
        except (FileNotFoundError, ValueError):
            self.create(img_list)
            self.save()