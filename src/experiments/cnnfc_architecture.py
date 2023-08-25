import sys

sys.path.append("./")

from src.config import CNNFCConfig, Config, FourierDatasetConfig, NetConfig, CNNConfig, ConvLayer
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "CNN_FC_architecture"

OPTIONS = [
    # [],
    # [64],
    # [256],
    # [512],
    # [1024],
    # [256, 128],
    [512, 256],
    [512, 256, 128],
    [512, 256, 128, 64],
]

def action(op, cfg: Config):
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = BEST_COMBINED_DATASET.__dict__
    
    cfg.net_config.net_class = "CNNFC"
    cfg.net_config.model_config = CNNFCConfig(
        input_size=300 + 16 + 1 + 1,
        output_size=5,
        cnn_input_size=300,
        in_channels=1,
        cnn_layers=BEST_CNN_CONFIG.conv_layers,
        fc_output_dim=BEST_FC_CONFIG.layers[-1],
        fc_layers=BEST_FC_CONFIG.layers[:-1],
        classifier_layers=op
    )
    
    cfg.net_config.name = f"CNNFC_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)