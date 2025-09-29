from abc import ABC, abstractmethod
import shutil
from rich.console import Console

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
from enum import Enum
import gc
import random


from utils.radiance_field_utils import RFModel, load_dataset, SH2RGB



class RadianceFieldEvalConfig():
    # radiance field models
    rf_config_paths: dict[str, Path] = {}
    
    # resolution scale factor
    res_factor: float = None
    
    # radiance field loading mode
    rf_load_mode: Literal["inference", "val", "test"] = "test"
    
    # dataset to load in radiance field
    rf_dset_mode: Literal["train", "test"] = "test"
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RadianceFieldEval(ABC):
    def __init__(
        self,
        cfg: RadianceFieldEvalConfig = RadianceFieldEvalConfig(),
    ):  
        # init console
        self._init_console()
        
        # config
        self.config = cfg
        
        # device
        self.device = self.config.device
        
        # init the radiance fields
        self._load_rf()
        
    def _init_console(self):
        """
        Console for logging/printing
        """
        # terminal width
        terminal_width = shutil.get_terminal_size((80, 20)).columns

        # rich console
        self.console = Console(width=terminal_width)
    
    def _load_rf(self):
        """
        Loads the radiance fields (RFs)
        """
        # radiance fields
        self.rad_field: dict = {}
        
        for scene, rf_path in tqdm(self.config.rf_config_paths.items(),
                                   desc="Loading the radiance fields"):
            # create dict
            self.rad_field[scene] = {}
            
            for rf_key, rf_cfg_path in rf_path.items():
                # load the RF
                self.rad_field[scene][rf_key] = RFModel(
                    config_path=Path(rf_cfg_path),
                    res_factor=self.config.res_factor,
                    test_mode=self.config.rf_load_mode,
                    dataset_mode=self.config.rf_dset_mode,
                    device=self.device,
                )
            
    @abstractmethod
    def evaluate_semantics_pca(self):
        """
        Runs Principal Component Analysis (PCA) for the semantic fields
        """
        pass
    
    @abstractmethod
    def evaluate_semantic_segmentation(self):
        """
        Evaluates semantic segmentation for the semantic fields
        """
        pass
    
    @abstractmethod
    def evaluate_rf_inversion(self):
        """
        Evaluates the radiance field inversion methods
        """
        pass
        
    