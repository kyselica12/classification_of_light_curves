import sys
import os

sys.path.append("./")

from src.config import  PACKAGE_PATH, FourierDatasetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *
from src.experiments.utils import get_default_cfg, load_dataset_to_trainer
from src.train import Trainer
from src.nn.networks.utils import get_new_net, load_net, save_net
from src.nn.datasets.fourier import FourierDataset

cfg = get_default_cfg()
cfg.data_config.dataset_class = "FourierDataset"
cfg.data_config.dataset_arguments = FourierDatasetConfig(fourier=True).__dict__
cfg.net_config.name = "FC_central_loss_165"
cfg.net_config.save_path = f"{PACKAGE_PATH}/output/models/Fourier_central_loss"

model_path = f"{PACKAGE_PATH}/output/models/{cfg.net_config.name}"
os.makedirs(model_path, exist_ok=True)
# model_path = cfg.net_config.save_path

cfg.net_config.net_class = "FC"
cfg.net_config.model_config = FCConfig(
    input_size=16,
    output_size=5,
    layers=[256,256,256]
)

trainer = Trainer(None, cfg.net_config, cfg.device)
trainer.central_loss = True
trainer.central_loss_weight = 1.65

load_dataset_to_trainer(trainer, f"Experiments_Fourier", cfg)
trainer.train_set.offset = 16
trainer.val_set.offset = 16
trainer.train_set.compute_std_mean()

print(FourierDataset.std, FourierDataset.mean)
mean, std = FourierDataset.mean, FourierDataset.std

trainer.net = get_new_net(cfg, f"{PACKAGE_PATH}/output/configurations/{cfg.net_config.name}.json")
trainer.add_sampler()

epochs = 300
save_interval = 50

report_path = f"{PACKAGE_PATH}/output/reports"
os.makedirs(report_path, exist_ok=True)

for i in range(0,epochs, save_interval):
    trainer.train(save_interval, 64, tensorboard_on=False, save_interval=None, print_on=False)
    save_net(trainer.net, "FC", model_path)
    trainer.performance_stats(cfg.data_config.labels, save_path=f"{report_path}/{cfg.net_config.name}_results.csv")


