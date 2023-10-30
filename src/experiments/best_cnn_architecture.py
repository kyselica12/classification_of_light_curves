import sys

sys.path.append("./")

from src.config import Config, FourierDatasetConfig, NetConfig, CNNConfig, ConvLayer
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "CNN_architecture_stride"

OPTIONS = [
    [(5, 11, 1)],
    [(5, 11, 3)],
    [(5, 11, 5)],
]

def action(op, cfg: Config):
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = FourierDatasetConfig(lc=True, push_to_max=True).__dict__
    
    cfg.net_config.net_class = "CNN"
    cfg.net_config.model_config = CNNConfig(
        input_size=300,
        output_size=5,
        conv_layers=op,
        classifier_layers=[256]
    )
    cfg.net_config.name = f"CNN_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)