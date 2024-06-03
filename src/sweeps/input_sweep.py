from copy import deepcopy
from src.sweeps.sweep import Sweep, DATA_CONFIG
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, NetArchitecture, CNNConfig, NetConfig, LC_SIZE, ResNetConfig
from src.configs import DataType as DT, AugmentType as A, SplitStrategy as ST

class InputSweep(Sweep):

    def get_data_cfg(self):
        return  DataConfig(
                path=f"{PACKAGE_PATH}/Fall_2021_csv",
                output_path=f"{PACKAGE_PATH}/resources/datasets",
                # validation_path = f"{PACKAGE_PATH}/resources/SDLCD.csv",
                class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
                regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
                validation_split=0.2,
                split_strategy=ST.TRACK_ID,
                number_of_training_examples_per_class=10_000,
                filter_config=FilterConfig( n_bins= 30, n_gaps= 10, gap_size= 5, rms_ratio= 0., non_zero_ratio=0.8),
                data_types=[],
                lc_shifts = 0,
                convert_to_mag=False,
                wavelet_scales= 10,
                wavelet_name= 'gaus1',
                train_augmentations=[],
        )

    def get_module_cfg(self):
        return NetConfig(
            name="CNN_sweep", input_size=LC_SIZE, 
            class_names=self.get_data_cfg().class_names,
            output_size=len(self.get_data_cfg().class_names),
            architecture=NetArchitecture.RESNET,
            learning_rate=0.001,
            args=ResNetConfig(n_layers=20, in_channels=1, output_size=5),
        )

    def get_wandb_sweep_cfg(self):
        return {
            "name": "Wawelet_sweep",
            "method": "random", 
            "metric": {
                "goal": "maximize",
                "name": "val_acc",
            },
            "parameters": {
                "wavelet_scales":{
                    "distribution": "int_uniform",   
                    "min": 5,
                    "max": 50,
                },
                "shifts": {
                    "values" : [True, False]
                },
                "label_smoothing": {
                    "values" : [True, False]
                },
                "mix_up": {
                    "values" : [True, False]
                },
                "num_epochs":{
                    "value": 30
                },
                "batch_size":{
                    "value": 32
                },
                "num_workers":{
                    "value": 4
                },
            }
        }

    def update_configs(self, config):

        data_cfg = self.get_data_cfg()
        module_cfg = self.get_module_cfg()

        data_types = []
        n_channels = 0


        print(module_cfg)

        for k,v in config.items():
            match [k, v]:
                case ["label_smoothing", True]:
                    module_cfg.label_smoothing = 0.1
                case ["mix_up", True]:
                    module_cfg.use_mixup = True
                    data_cfg.train_augmentations.append(A.MIX_UP)
                case ["shifts", True]:
                    data_cfg.train_augmentations.append(A.SHIFT)
                case ["wavelet_scales", n]:
                    data_cfg.wavelet_scales = n
                    data_types.append(DT.WAVELET)
                    n_channels += n

        data_cfg.data_types = data_types
        module_cfg.args.in_channels = n_channels

        return data_cfg, module_cfg