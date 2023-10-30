import sys

sys.path.append("./")

from src.config import  PACKAGE_PATH, FourierDatasetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *
from src.experiments.utils import get_default_cfg, load_dataset_to_trainer
from src.train import Trainer
from src.nn.networks.utils import get_new_net, load_net, save_net


cfg = get_default_cfg()
cfg.data_config.dataset_class = "FourierDataset"
cfg.data_config.dataset_arguments = FourierDatasetConfig(fourier=True).__dict__
cfg.net_config.name = "FC_trained"
cfg.net_config.save_path = f"{PACKAGE_PATH}/output/models/Fourier_trained"

model_path = f"{PACKAGE_PATH}/output/models/Fourier_trained"

cfg.net_config.net_class = "FC"
cfg.net_config.model_config = FCConfig(
    input_size=16,
    output_size=5,
    layers=[256,256,256]
)

trainer = Trainer(None, cfg.net_config, cfg.device)

load_dataset_to_trainer(trainer, f"Experiments", cfg)

trainer.net = get_new_net(cfg, f"{PACKAGE_PATH}/output/configurations/{cfg.net_config.name}.json")
trainer.add_sampler()

epochs = 300    
save_interval = 10

for i in range(0,epochs, save_interval):
    trainer.train(save_interval, 64, tensorboard_on=False, save_interval=None, print_on=False)
    save_net(trainer.net, cfg.net_config.name, model_path)
    trainer.performance_stats(cfg.data_config.labels, save_path=f"{PACKAGE_PATH}/output/Fourier_trained.csv")


