from copy import deepcopy
from src.sweeps.sweep import Sweep, DATA_CONFIG
from src.configs import NetArchitecture, CNNConfig, NetConfig, LC_SIZE
from src.configs import DataType as DT

class WaveletSweep(Sweep):

    def get_data_cfg(self):
        data_cfg = deepcopy(DATA_CONFIG)
        data_cfg.data_types = [DT.LC]
        return data_cfg

    def get_module_cfg(self):
        return NetConfig(
            name="CNN_sweep", input_size=LC_SIZE, 
            class_names=self.get_data_cfg().class_names,
            output_size=len(self.get_data_cfg().class_names),
            architecture=NetArchitecture.CNN,
            learning_rate=0.001,
            args=CNNConfig(input_size=LC_SIZE, output_size=5, in_channels=1, 
                        conv_layers=[(16,7, 5),(32, 3, 1)], 
                        classifier_layers=[])
        )

    def get_wandb_sweep_cfg(self):
        return {
            "name": "Wawelet_sweep",
            "method": "random", 
            "metric": {
                "goal": "minimize",
                "name": "val_loss_epoch",
            },
            "parameters": {
                "recontructed": {
                    "values": [True, False]
                },
                "lc_shifts": {
                    "values" : [0, 1]
                },
                "wavelet_scales":{
                    # "distribution": "int_uniform",   
                    # "min": 10,
                    # "max": 100,
                    "value": 10
                },
                "num_epochs":{
                    "value": 200
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
                case ["lc_shifts", n]:
                    if n > 0:
                        data_cfg.lc_shifts = n-1
                        data_types.append(DT.LC)
                        n_channels += 1
                case ["recontructed", True]:
                    data_types.append(DT.RECONSTRUCTED_LC)
                    n_channels += 1
                case ["wavelet_scales", n]:
                    data_cfg.wavelet_scales = n
                    data_types.append(DT.WAVELET)
                    n_channels += n

        data_cfg.data_types = data_types
        module_cfg.args.in_channels = n_channels

        return data_cfg, module_cfg