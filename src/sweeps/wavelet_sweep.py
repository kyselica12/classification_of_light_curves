from copy import deepcopy
from src.sweeps.sweep import Sweep, DATA_CONFIG
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, NetArchitecture, CNNConfig, NetConfig, LC_SIZE, ResNetConfig
from src.configs import DataType as DT, AugmentType as A, SplitStrategy as ST

class WaveletSweep(Sweep):

    def get_data_cfg(self):
        data_cfg = DataConfig(
                path=f"{PACKAGE_PATH}/Fall_2021_csv",
                output_path=f"{PACKAGE_PATH}/resources/datasets",
                class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
                regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
                validation_split=0.2,
                split_strategy=ST.TRACK_ID,
                number_of_training_examples_per_class=100_000,
                filter_config=FilterConfig(n_bins=30, n_gaps= 10, gap_size=5, rms_ratio= 0., non_zero_ratio=0.8),
                # filter_config=None,
                data_types=[DT.WAVELET],
                wavelet_start_scale=1,
                wavelet_scales_step=1,
                wavelet_end_scale=5,
                wavelet_name= 'gaus1',
                lc_shifts = 0,
                convert_to_mag=False,
                train_augmentations=[A.SHIFT],
        )
        return data_cfg
        # return  DataConfig(
        #         path=f"{PACKAGE_PATH}/Fall_2021_csv",
        #         output_path=f"{PACKAGE_PATH}/resources/datasets",
        #         # validation_path = f"{PACKAGE_PATH}/resources/SDLCD.csv",
        #         class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
        #         regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        #         validation_split=0.2,
        #         split_strategy=ST.TRACK_ID,
        #         number_of_training_examples_per_class=10_000,
        #         filter_config=FilterConfig( n_bins= 30, n_gaps= 10, gap_size= 5, rms_ratio= 0., non_zero_ratio=0.8),
        #         data_types=[DT.WAVELET],
        #         lc_shifts = 0,
        #         convert_to_mag=False,
        #         wavelet_name= 'gaus1',
        #         wavelet_end_scale=5,
        #         wavelet_scales_step=1,
        #         wavelet_start_scale=1,
        #         train_augmentations=[],
        # )

    def get_module_cfg(self):
        return NetConfig(
            name="CNN_sweep", input_size=LC_SIZE, 
            class_names=self.get_data_cfg().class_names,
            output_size=len(self.get_data_cfg().class_names),
            architecture=NetArchitecture.RESNET,
            learning_rate=0.001,
            args=ResNetConfig(n_layers=20, in_channels=5, output_size=5),
        )

    def get_wandb_sweep_cfg(self):
        return {
            "name": "Wavelet_type2",
            "method": "grid", 
            "metric": {
                "goal": "maximize",
                "name": "val_acc_0",
            },
            "parameters": {
                "wavelet_name":{
                    "values": ["gaus1", "gaus2", "morl", "mexh"],
                },
                "num_epochs":{
                    "value": 40
                },
                "batch_size":{
                    "value": 32
                },
                "num_workers":{
                    "value": 0
                },
            }
        }

    def update_configs(self, config):

        data_cfg = self.get_data_cfg()
        module_cfg = self.get_module_cfg()

        print(module_cfg)

        for k,v in config.items():
            match [k, v]:
                case ["wavelet_name", name]:
                    data_cfg.wavelet_name = name 

        return data_cfg, module_cfg
