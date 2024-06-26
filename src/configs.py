from collections import namedtuple
import numpy as np
from dataclasses import dataclass, field
from strenum import StrEnum
from typing import Dict, List, Tuple

PACKAGE_PATH = "/home/k/kyselica12/work/classification_of_light_curves"
# PACKAGE_PATH = "/home/poirot/work/classification_of_light_curves"

WANDB_KEY_FILE = f"{PACKAGE_PATH}/wandb.key"
LC_SIZE = 300
FOURIER_N = 8


RB_NAMES =   ["CZ-3B", "Falcon 9", "Atlas 5 Centaur",  "H2-A", "Ariane 5", "Delta 4", "Titan 3"]
RB_REGEXES = [r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R.B$',  
             r'H-2A.*', r'ARIANE_5.*', r'DELTA_4.*', r'TITAN_3C_TRANSTAGE.*']

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

@dataclass
class ResNetConfig(ModelConfig):
    n_layers: int = 20
    in_channels: int = 1

class NetArchitecture(StrEnum):
    FC = "FullyConnected"
    CNN = "CNN"
    CNNFC = "CNNFC"
    RESNET = "ResNet"

@dataclass
class NetConfig:
    name: str = "DEFAULT"
    input_size: int = 300
    output_size: int = 5
    learning_rate: float = 1e-3
    label_smoothing: float = 0.
    use_mixup: bool = False
    class_names: List[str] = field(default_factory=list)
    architecture: NetArchitecture = NetArchitecture.FC
    args: ModelConfig = None
    sweep: bool = False


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
    NO_SPLIT = "no_split"

class DataType(StrEnum):
    LC = "light_curve" # LC_SIZE
    FS = "fourier_series_coefs" # FOURIER_N x (n_shifts+1)
    STD = "fourier_series_std" # FOURIER_N
    RECONSTRUCTED_LC = "reconstructed_lc" # LC_SIZE
    RESIDUALS = "residuals" # LC_SIZE
    RMS = "rms" # 1
    AMPLITUDE = "amplitude" # 1
    WAVELET = "wavelet_transform" # LC_SIZE x wavelet_scales

class DatasetType(StrEnum):
    TRAIN = "train"
    TEST = "test"
    ARTIFICIAL = "artificial"

class AugmentType(StrEnum):
    SHIFT = "shift"
    INSERT_GAP = "insert_gap"
    MIX_UP = "mix_up"
    NOISE = "noise"

MIX_UP_ALPHA = 0.2

@dataclass 
class DataConfig:
    path: str = ""
    validation_path: str = ""
    artificial_data_path: str = None
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

    wavelet_name: str = "morl"
    wavelet_scales_step: int = 1
    wavelet_start_scale: int = 1
    wavelet_end_scale: int = 100

    lc_shifts: int = 0

    data_types: List[DataType | Tuple] = field(default_factory=list)
    train_augmentations: List[AugmentType] = None
