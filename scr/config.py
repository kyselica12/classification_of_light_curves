import yaml
from dataclasses import dataclass
from typing import Tuple, Union

@dataclass
class DataConfig:
    path: str
    labels: dict

@dataclass
class NetConfig:
    name: str
    input_size: int
    n_channels: int
    n_classes: int
    hid_dim: int
    stride: int
    kernel_size: int
    checkpoint: Union[str, int] = None
    device: str = "cpu"

@dataclass
class FilterConfig:
    n_bins: int
    n_gaps: int
    gap_size: int
    non_zero_ratio: float
    rms_ratio: float = 0


def load_config() -> Tuple[DataConfig, NetConfig, FilterConfig]:
    PATH = "C:\\Users\\danok\\work\\dizertacka\\classification_of_light_curves\\scr\\comfig.yaml"
    
    with open(PATH, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    data_cfg = DataConfig(path=cfg["data"]['path'], labels=cfg["data"]["labels"])
    net_cfg = NetConfig(
                        name=cfg["net"]["name"],
                        checkpoint=cfg["net"]["checkpoint"],
                        input_size=cfg["net"]["input_size"],
                        n_channels=cfg["net"]["n_channels"],
                        hid_dim=cfg["net"]["hid_dim"],
                        stride=cfg["net"]["stride"],
                        kernel_size=cfg["net"]["kernel_size"],
                        n_classes=cfg["net"]["n_classes"]
            )
    filter_cfg = FilterConfig(n_bins=cfg["filter"]["n_bins"],
                              n_gaps=cfg["filter"]["n_gaps"],
                              gap_size=cfg["filter"]["gap_size"],
                              non_zero_ratio=cfg["filter"]["non_zero_ratio"])

    
    return data_cfg, net_cfg, filter_cfg