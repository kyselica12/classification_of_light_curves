from copy import deepcopy
from src.sweeps.sweep import Sweep, DATA_CONFIG
from src.configs import NetArchitecture, CNNConfig, NetConfig, LC_SIZE, DataConfig, FilterConfig, PACKAGE_PATH
from src.configs import SplitStrategy as ST, AugmentType as A, DataType as DT

class ResnetPreprocesingSweep(Sweep):

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
            name="Resnet20", input_size=LC_SIZE, 
            class_names=self.get_data_cfg().class_names,
            output_size=len(self.get_data_cfg().class_names),
            architecture=NetArchitecture.RESNET20,
            learning_rate=0.001,
            args={},
        )

    def get_wandb_sweep_cfg(self):
        return {
            "name": "Preprocessing",
            "method": "grid", 
            "metric": {
                "goal": "maximize",
                "name": "val_acc",
            },
            "parameters": {
                "data_type": {
                    "values": [DT.LC, DT.RECONSTRUCTED_LC]
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
                    "value": 25
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

        print(module_cfg)

        for k,v in config.items():
            match [k, v]:
                case ["data_type", dt]:
                    data_cfg.data_types = [dt]
                case ["label_smoothing", True]:
                    module_cfg.label_smoothing = 0.1
                case ["mix_up", True]:
                    module_cfg.use_mixup = True
                    data_cfg.train_augmentations.append(A.MIX_UP)
                case ["shifts", True]:
                    data_cfg.train_augmentations.append(A.SHIFT)


        return data_cfg, module_cfg