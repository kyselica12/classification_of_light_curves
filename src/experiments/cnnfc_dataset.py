import sys

sys.path.append("./")

from src.config import CNNFCConfig, Config, FourierDatasetConfig, NetConfig, CNNConfig, ConvLayer
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "CNN_FC_architecture"

lc = ["lc"]
reconstructed_lc = ["reconstructed_lc"]
residuals = ["residuals"]

OPTIONS = [
    lc,
    reconstructed_lc,
    residuals,
    lc + reconstructed_lc,
    lc + residuals,
    reconstructed_lc + residuals,
    lc + reconstructed_lc + residuals,
]

def action(op, cfg: Config):
    
    dataset_cfg = BEST_COMBINED_DATASET
    
    cfg.net_config.net_class = "CNNFC"
    cfg.net_config.model_config = BEST_CNN_FC_CONFIG
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = FourierDatasetConfig(
                                            fourier=True,
                                            amplitude=True,
                                            rms=True,
                                            lc="lc" in op,
                                            reconstructed_lc="reconstructed_lc" in op,
                                            residuals="residuals" in op,
                                            push_to_max=True,
                                        ).__dict__
        
    channels = len(op)
    cfg.net_config.model_config.in_channels = channels
    cfg.net_config.model_config.input_size = 300 * channels + 16 + 1 + 1
        
    cfg.data_config.dataset_class = "FourierDataset"
    
    cfg.net_config.name = f"CNNFC_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)