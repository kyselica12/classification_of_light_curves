import sys

sys.path.append("./")

from src.config import Config, FourierDatasetConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "FC_dataset"

fourier = ["fourier"]
amp = ["amp"]
rms = ["rms"]

OPTIONS = [
    fourier,
    fourier + amp,
    fourier + rms,
    fourier + amp + rms,
]

def action(op, cfg: Config):
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = FourierDatasetConfig(
        fourier="fourier" in op,
        amplitude="amp" in op,
        rms="rms" in op
    ).__dict__
    
    cfg.net_config.net_class = "FC"
    cfg.net_config.model_config = BEST_FC_CONFIG
    cfg.net_config.model_config.input_size=16 + (1 if "amp" in op else 0) + (1 if "rms" in op else 0)
       
    cfg.net_config.name = f"FC_dataset_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)