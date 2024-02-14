import numpy as np
from dataclasses import dataclass, field
from strenum import StrEnum
from typing import List


PACKAGE_PATH = "/home/poirot/work/classification_of_light_curves"
WANDB_KEY_FILE = "wandb.key"
LC_SIZE = 300
FOURIER_N = 8

class NetArchitecture(StrEnum):
    FC = "FullyConnected"
    CNN = "CNN"

@dataclass
class NetConfig:
    name: str = "DEFAULT"
    input_size: int = 300
    output_size: int = 5
    architecture: NetArchitecture = NetArchitecture.FC
    args: dict = field(default_factory=dict)


@dataclass
class FilterConfig:
    n_bins: int = 30
    n_gaps: int = 2
    gap_size: int = 1
    non_zero_ratio: float = 0.8
    rms_ratio: float = 0. 

@dataclass 
class DataConfig:
    path: str = ""
    output_path: str = None
    class_names: List[str] = field(default_factory=list)
    regexes: List[str] = field(default_factory=list)
    validation_split: float = 0.2
    number_of_training_examples_per_class: int = 10_000
    filter_config: FilterConfig = field(default_factory=lambda: FilterConfig())
    convert_to_mag: bool = False
    max_amplitude: float = 20
    args: dict = field(default_factory=dict)
    fourier: bool = False
    std: bool = False
    rms: bool = False
    residuals: bool = False
    amplitude: bool = False
    lc: bool = False
    reconstructed_lc: bool = False
    push_to_max: bool = False
 
