
from dataclasses import dataclass
from typing import List
from dataclasses_json import dataclass_json
import numpy as np

PACKAGE_PATH = "C:/Users/Kyselica/Desktop/kyselica/classification_of_light_curves"

@dataclass_json
@dataclass
class NetConfig:
    name: str = "Real_Augmented_5"
    input_size: int = 300
    n_channels: int = 10
    n_classes: int = 5
    hid_dim: int = 128
    stride: int = 2
    kernel_size: int = 5
    device: str = "cuda:0"
    checkpoint: int = None
    save_path: str = "C:/Users/Kyselica/Desktop/kyselica/classification_of_light_curves/resources/models"

@dataclass_json
@dataclass
class FilterConfig:
    n_bins: int = 30
    n_gaps: int = 2
    gap_size: int = 1
    non_zero_ratio: float = 0.2
    rms_ratio: float = 0.

@dataclass_json
@dataclass
class AugmentationConfig:
    min_examples: int = 1000
    roll: bool = True 
    add_gaps: bool = True 
    add_noise: bool = True
    max_noise: float = .03
    keep_original: bool = True
    num_gaps: int = 1
    gap_prob: float = .2
    min_gap_len: int = 1
    max_gap_len: int = 3

@dataclass_json
@dataclass
class DataConfig:
    path: str = ""
    batch_size = 128
    convert_to_mag: bool = True
    labels: List[str] = None
    regexes: List[str] = None
    validation_split: float = 0.1
    filter: FilterConfig = None
    augmentation: AugmentationConfig = None
    save_path: str = None
    number_of_training_examples_per_class: int = np.inf
    
@dataclass_json
@dataclass
class Config:
    net_config: NetConfig = NetConfig()
    data_config: DataConfig = DataConfig()
    seed: int = 0

    