from collections import namedtuple
import numpy as np
from dataclasses import dataclass, field
from strenum import StrEnum
from typing import List, Tuple


PACKAGE_PATH = "/home/poirot/work/classification_of_light_curves"
WANDB_KEY_FILE = "wandb.key"
LC_SIZE = 300
FOURIER_N = 8

@dataclass
class ModelConfig:
    input_size: int = 300
    output_size: int = 5

ConvLayer = namedtuple("ConvLayer", ["out_ch", "kernel", "stride"], defaults=[1,3,1])
@dataclass
class CNNConfig(ModelConfig):
    in_channels: int = 1
    conv_layers: List[ConvLayer] = field(default_factory=list)
    classifier_layers: List[int] = field(default_factory=list)

@dataclass
class FCConfig(ModelConfig):
    layers: List[int] = field(default_factory=list)

@dataclass
class CNNFCConfig(ModelConfig):
    in_channels: int = 1
    cnn_input_size: int = 0
    cnn_layers: List[Tuple] = field(default_factory=list)
    fc_output_dim: int = 10
    fc_layers: List[int] = field(default_factory=list)
    classifier_layers: List[int] = field(default_factory=list)

class NetArchitecture(StrEnum):
    FC = "FullyConnected"
    CNN = "CNN"
    CNNFC = "CNNFC"

@dataclass
class NetConfig:
    name: str = "DEFAULT"
    input_size: int = 300
    output_size: int = 5
    class_names: List[str] = field(default_factory=list)
    architecture: NetArchitecture = NetArchitecture.FC
    args: ModelConfig = None


@dataclass
class FilterConfig:
    n_bins: int = 30
    n_gaps: int = 2
    gap_size: int = 1
    non_zero_ratio: float = 0.8
    rms_ratio: float = 0. 

class SplitStrategy(StrEnum):
    RANDOM = "random"
    OBJECT_ID = "objectID"
    TRACK_ID = "trackID"

@dataclass 
class DataConfig:
    path: str = ""
    output_path: str = None

    class_names: List[str] = field(default_factory=list)
    regexes: List[str] = field(default_factory=list)
    validation_split: float = 0.2
    split_strategy: str = SplitStrategy.RANDOM
    seed: int = 42
    number_of_training_examples_per_class: int = 10_000
    filter_config: FilterConfig = field(default_factory=lambda: FilterConfig())
    convert_to_mag: bool = False
    max_amplitude: float = 20

    fourier: bool = False
    std: bool = False
    rms: bool = False
    residuals: bool = False
    amplitude: bool = False
    lc: bool = False
    reconstructed_lc: bool = False
    push_to_max: bool = False
    wavelet: bool = False
    wavelet_scales: int = 30
    wavelet_name: str = "morl"
 