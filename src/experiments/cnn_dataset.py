import sys

sys.path.append("./")

from src.config import Config, FourierDatasetConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *

EXPERIMENT_NAME = "CNN_dataset_size"

OPTIONS = [
    1000,
    2000,
    3000,
    10_000,
]

def action(op, cfg: Config):
    
    cfg.data_config.dataset_class = "FourierDataset"
    cfg.data_config.dataset_arguments = BEST_CNN_DATASET.__dict__
    cfg.data_config.number_of_training_examples_per_class = op
    
    cfg.net_config.net_class = "CNN"
    cfg.net_config.model_config = BEST_CNN_CONFIG
    
    cfg.net_config.name = f"DatasetSize_{op}"

run_experiment(
    options=OPTIONS,
    action=action,
    name=EXPERIMENT_NAME
)