
from ast import Tuple
from dataclasses import field, dataclass
from typing import List
from dataclasses_json import dataclass_json
from collections import namedtuple
import numpy as np



# PACKAGE_PATH = "D:/work/classification_of_light_curves"
PACKAGE_PATH = "/media/bach/DATA/work/classification_of_light_curves"

@dataclass_json
@dataclass
class ModelConfig:
    input_size: int = 300
    output_size: int = 5

ConvLayer = namedtuple("ConvLayer", ["out_ch", "kernel", "stride"], defaults=[1,3,1])
@dataclass_json
@dataclass
class CNNConfig(ModelConfig):
    in_channels: int = 1
    conv_layers: List[ConvLayer] = field(default_factory=list)
    classifier_layers: List[int] = field(default_factory=list)

@dataclass_json
@dataclass
class FCConfig(ModelConfig):
    layers: List[int] = field(default_factory=list)

@dataclass_json
@dataclass
class CNNFCConfig(ModelConfig):
    in_channels: int = 1
    cnn_input_size: int = 0
    cnn_layers: List[Tuple] = field(default_factory=list)
    fc_output_dim: int = 10
    fc_layers: List[int] = field(default_factory=list)
    classifier_layers: List[int] = field(default_factory=list)

@dataclass_json
@dataclass
class NetConfig:
    name: str = "Real_Augmented_5"
    save_path: str = f"{PACKAGE_PATH}/resources/models"
    net_class: str = "Net"
    model_config: ModelConfig = None

@dataclass_json
@dataclass
class FilterConfig:
    n_bins: int = 30
    n_gaps: int = 2
    gap_size: int = 1
    non_zero_ratio: float = 0.8
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
class FourierDatasetConfig:
    fourier: bool = False
    std: bool = False
    residuals: bool = False
    rms: bool = False
    amplitude: bool = False
    lc: bool = False
    reconstructed_lc: bool = False
    push_to_max: bool = True

@dataclass_json
@dataclass
class DataConfig:
    path: str = ""
    batch_size: int = 128
    convert_to_mag: bool = True
    labels: List[str] = None
    regexes: List[str] = None
    validation_split: float = 0.1
    filter: FilterConfig = None
    dataset_class: str = "BasicDataset"
    dataset_arguments: dict = field(default_factory=dict)
    save_path: str = None
    number_of_training_examples_per_class: int = np.inf
    
@dataclass_json
@dataclass
class Config:
    net_config: NetConfig = field(default_factory=lambda: NetConfig())
    device: str = "cuda:0"
    data_config: DataConfig = field(default_factory=lambda: DataConfig())
    seed: int = 0


    