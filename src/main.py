import sys
import os
from copy import deepcopy

sys.path.append("..")
sys.path.append(".")

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb

from src.configs import DataConfig, NetConfig, NetArchitecture, FilterConfig, DataType, CNNConfig, LC_SIZE, ResNetConfig
from src.configs import AugmentType as A, DatasetType as DST, RB_NAMES, RB_REGEXES, DatasetType as DST
from src.utils import train, get_wandb_logger
from src.configs import PACKAGE_PATH
from src.data.data_processor import DataProcessor
from src.sweeps.sweep import DATA_CONFIG
from src.module.lightning_module import LCModule
from src.configs import SplitStrategy as ST
from src.module.resnet import resnet20
from src.utils import get_wandb_logger

import torch
import numpy as np
import random

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    data_cfg = DataConfig(
            path=f"{PACKAGE_PATH}/Fall_2021_csv",
            output_path=f"{PACKAGE_PATH}/resources/datasets",
            # validation_path = f"{PACKAGE_PATH}/resources/SDLCD(2).csv",
            # artificial_data_path=f"{PACKAGE_PATH}/artificial_hist",
            class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
            regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
            # class_names=RB_NAMES[:4],
            # regexes=RB_REGEXES[:4],
            validation_split=0.2,
            split_strategy=ST.TRACK_ID,
            number_of_training_examples_per_class=100_000,
            filter_config=FilterConfig(n_bins=30, n_gaps= 10, gap_size=5, rms_ratio= 0., non_zero_ratio=0.8),
            # filter_config=None,
            data_types=[DataType.WAVELET],
            wavelet_start_scale=1,
            wavelet_scales_step=1,
            wavelet_end_scale=5,
            wavelet_name= 'gaus1',
            lc_shifts = 0,
            convert_to_mag=False,
            train_augmentations=[A.SHIFT],
    )


    net_cfg = NetConfig(
        name="Resnet20", input_size=LC_SIZE, 
        class_names=data_cfg.class_names,
        output_size=len(data_cfg.class_names),
        architecture=NetArchitecture.RESNET,
        args=ResNetConfig(n_layers=20, in_channels=5, output_size=5),
        learning_rate=0.001,
        label_smoothing=0.1,
    )




    dp = DataProcessor(data_cfg)
    print("LOAD")


    # dp.create_dataset_from_csv(type=DST.ARTIFICIAL)
    # dp.save_data(type=DST.ARTIFICIAL)
    # dp.save_data()

    if os.path.exists(f'{dp.output_path}/{dp.hash}'):
        dp.load_data_from_file(DST.TRAIN)
        # dp.load_data_from_file(DST.ARTIFICIAL)
        dp.load_data_from_file(type=DST.TEST)
    else:
        dp.create_dataset_from_csv(DST.TRAIN)
        # dp.create_dataset_from_csv(DST.ARTIFICIAL)
        dp.create_dataset_from_csv(DST.TEST)
        dp.save_data(DST.TRAIN)
        # dp.save_data(DST.ARTIFICIAL)
        dp.save_data(DST.TEST)
    

    PROJECT = "4 Rocket Bodies"
    NAME = "Wavelet best"
    # logger = None
    print("WANDB")
    wandb.init(project=PROJECT, name=NAME)
    logger = get_wandb_logger(PROJECT,NAME)

    print("MODULE")
    net_cfg.sweep = False
    module: LCModule = LCModule(net_cfg)
    # module.log_confusion_matrix = True
    module.log_confusion_matrix = False

    os.makedirs(f"{PACKAGE_PATH}/resources/models/{PROJECT}/{NAME}", exist_ok=True)

    print("TRAIN")

    train(module=module,
        dp=dp,
        num_epochs=50,
        batch_size=32,
        num_workers=0,
        callbacks = [ModelCheckpoint(monitor='val_acc_0', mode='max', save_top_k=1,dirpath=f"{PACKAGE_PATH}/resources/models/{PROJECT}/{NAME}")],
        sampler=True,
        max_num_samples=10_000,
        logger=logger)

    print("DONE")

