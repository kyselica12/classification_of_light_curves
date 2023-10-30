import sys

sys.path.append("./")

from src.config import Config, DataConfig, FilterConfig, PACKAGE_PATH, FourierDatasetConfig, NetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "FC_architecture"

OPTIONS = [
    [],
    [128],
    [1024],
    [128, 256],
    [128,256,512],
    [128,256,256],
    [256,256,256],
    [256,256,256, 256]
]

def action(op, cfg: Config):
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = FourierDatasetConfig(fourier=True).__dict__
    
    cfg.net_config.net_class = "FC"
    cfg.net_config.model_config = FCConfig(
        input_size=16,
        output_size=5,
        layers=op
    )
    cfg.net_config.name = f"FC_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)