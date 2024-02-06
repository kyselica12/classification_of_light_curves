import numpy as np
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List


PACKAGE_PATH = "/media/bach/DATA/work/classification_of_light_curves"

@dataclass
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
    labels: List[str] = field(default_factory=list)
    regexes: List[str] = field(default_factory=list)
    validation_split: float = 0.2
    number_of_training_examples_per_class: int = np.inf
    filter_config: FilterConfig = FilterConfig()
    args: dict = field(default_factory=dict)

