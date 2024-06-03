from copy import deepcopy
from abc import ABC, abstractmethod
import os

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


from src.utils import train
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, NetConfig
from src.configs import DataType as DT
from src.data.data_processor import DataProcessor
from src.module.lightning_module import LCModule

DATA_CONFIG = DataConfig(
        path=f"{PACKAGE_PATH}/Fall_2021_csv",
        output_path=f"{PACKAGE_PATH}/resources/datasets",
        class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        validation_split=0.2,
        split_strategy="objectID",
        number_of_training_examples_per_class=10_000,
        filter_config=FilterConfig( n_bins= 30, n_gaps= 10, gap_size= 5, rms_ratio= 0., non_zero_ratio=0.8),
        data_types=[],
        lc_shifts = 0,
        convert_to_mag=False,
        wavelet_name= 'gaus1',
        train_augmentations=None
)

class Sweep:

    @abstractmethod
    def get_data_cfg(self) -> DataConfig:
        pass

    @abstractmethod
    def get_module_cfg(self) -> NetConfig:
        pass

    @abstractmethod
    def update_configs(self) -> tuple[DataConfig, NetConfig]:
        pass

    @abstractmethod
    def get_wandb_sweep_cfg(self):
        pass

    def run(self):
        with wandb.init():
            config = wandb.config

            data_cfg, module_cfg = self.update_configs(config)

            dp = DataProcessor(data_cfg)

            if os.path.exists(f'{dp.output_path}/{dp.hash}'):
                dp.load_data_from_file()
            else:
                dp.create_dataset_from_csv()
                dp.save_data()
            
            print("Data loaded")

            module_cfg.input_size = dp.data_shape()
            module_cfg.sweep = False
            module = LCModule(module_cfg)
            module.log_confusion_matrix = False

            logger = WandbLogger()
            # logger = None
            print("Testing string")

            train(module, dp,
                num_epochs=config["num_epochs"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                callbacks=[],
                # callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=5)],
                logger=logger)