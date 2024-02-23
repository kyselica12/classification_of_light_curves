import sys
from copy import deepcopy

import wandb
from pytorch_lightning.loggers import WandbLogger

sys.path.append(".")


from src.utils import get_wandb_logger, train, log_in_to_wandb
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, ModelConfig, NetArchitecture, FCConfig, CNNConfig, NetConfig, LC_SIZE
from src.configs import DataType as DT
from src.data.data_processor import DataProcessor
from src.module.lightning_module import LCModule

PROJECT = "SWEEP_CNN"

DATA_CFG = DataConfig(
    path="/mnt/c/work/Fall_2021_csv",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
    regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
    validation_split=0.2,
    split_strategy="objectID",
    number_of_training_examples_per_class=10_000,
    filter_config=FilterConfig( n_bins= 30, n_gaps= 10, gap_size= 5, rms_ratio= 0., non_zero_ratio=0.8),
    data_types=[DT.LC],
    lc_shifts = 0,
    convert_to_mag=False,
    wavelet_scales= 10,
    wavelet_name= 'gaus1',
    train_augmentations=None
)

MODULE_CFG = NetConfig(
    name="CNN_sweep", input_size=LC_SIZE, 
    class_names=DATA_CFG.class_names,
    output_size=5,
    architecture=NetArchitecture.CNN,
    args=CNNConfig(input_size=LC_SIZE, output_size=5, in_channels=1, 
                   conv_layers=[(16, 3, 1), (32, 3, 1)], 
                   classifier_layers=[])
)

def main(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        module_cfg = deepcopy(MODULE_CFG)
        module_cfg.learning_rate = cfg.learning_rate
        module_cfg.args.conv_layers = cfg.conv_layers
        module_cfg.args.classifier_layers = cfg.head_layers

        dp = DataProcessor(DATA_CFG)

        dp.load_data_from_file()
        module = LCModule(module_cfg)
        logger = WandbLogger(project=PROJECT, log_model=False)

        train(module, dp,
            num_epochs=100,
            batch_size=cfg.batch_size,
            num_workers=4,
            logger=logger)


sweep_config = {
    "method": "grid", 
    "metric": {
        "goal": "minimize",
        "name": "val_loss",
    },
    "parameters": {
        "batch_size": {
            "values": [32, 64, 128, 256],
        },
        "learning_rate": {
            "values": [0.1, 0.01, 0.001, 0.0001]
        },
        "conv_layers": {
            "values": [
                [(16, 3,1)],
                [(16, 7,3)],
                [(16, 11,5)],
                [(16, 3,1), (16, 3, 1)],
                [(16, 3,1), (16, 3, 1), (16, 3, 1)],
                [(16, 7,3), (16, 3, 1), (16, 3, 1)],
                [(16, 11,5), (16, 3, 1), (16, 3, 1)],
            ]
        },
        "head_layers":{
            "values": [
                [128],
                [256],
                [512],
                [1024],
                [256,256],
                [256,256,256],
                [256,256,256, 256]
            ]
        }
    }
}

log_in_to_wandb()

sweep_id = wandb.sweep(sweep_config, project=PROJECT)    
wandb.agent(sweep_id, function=main)

    