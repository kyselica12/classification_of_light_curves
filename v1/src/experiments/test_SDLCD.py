import sys

sys.path.append("./")

from src.config import  PACKAGE_PATH, FourierDatasetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *
from src.experiments.utils import get_default_cfg, load_dataset_to_trainer
from src.main import Trainer
from src.nn.networks.utils import get_new_net, load_net, save_net
from src.nn.networks.fc import FC
from src.experiments.utils import test_SDLCD


cfg = get_default_cfg()
cfg.device = "cpu"
cfg.data_config.dataset_class = "FourierDataset"
cfg.data_config.dataset_arguments = FourierDatasetConfig(fourier=True).__dict__
cfg.net_config.name = "FC_trained"
cfg.net_config.save_path = f"{PACKAGE_PATH}/output/models/Fourier_trained"


cfg.net_config.net_class = "FC"
cfg.net_config.model_config = FCConfig(
    input_size=16,
    output_size=5,
    layers=[256,256,256]
)
# cfg.net_config.name = "Finetuned_SDLCD"


SDLCD_data_path = f"{PACKAGE_PATH}/output/datasets/SDLCD"
SEED = 96752
EPOCH = 10
train_dataset_folder = "Experiments"



# test_SDLCD(SDLCD_data_path, cfg, seed=SEED, epoch=EPOCH, checkpoint=0, train_dataset_folder=train_dataset_folder, output_path=".")

from src.nn.datasets.fourier import FourierDataset
import numpy as np
import torch
EPOCHS = 30
model_dir = f"{PACKAGE_PATH}/output/models/Fourier_trained"
model_path = f"{model_dir}/FC_epochs_{EPOCHS}_checkpoint_000.model"

net = FC(cfg.net_config.model_config)
net.load_state_dict(torch.load(model_path))
net.to(cfg.device)
net.double()

trainer = Trainer(net, cfg.net_config, device=cfg.device)

load_dataset_to_trainer(trainer, f"Experiments_Fourier", cfg)
trainer.train_set.offset = 16
trainer.val_set.offset = 16
trainer.train_set.compute_std_mean()
mean, std = FourierDataset.mean, FourierDataset.std


# test_set = FourierDataset([],[],**cfg.data_config.dataset_arguments)
# test_set.data = np.load(f"{SDLCD_data_path}/test_x.npy")
# test_set.labels = np.load(f"{SDLCD_data_path}/test_y.npy").astype(dtype=np.int32)
# test_set.offset = 16    
# test_set.compute_std_mean()
# mean, std = test_set.mean, test_set.std
# test_set.mean = mean
# test_set.std = std
# trainer.val_set = test_set


trainer.load_data_from_file(SDLCD_data_path, cfg.data_config)
trainer.train_set.offset = 16
trainer.val_set.offset = 16
trainer.train_set.compute_std_mean()
# trainer.train_set.mean = mean
# trainer.train_set.std = std
# trainer.val_set.mean = mean
# trainer.val_set.std = std


trainer.performance_stats(cfg.data_config.labels)

trainer.add_sampler()

epochs = 100    
save_interval = 5

for i in range(0,epochs, save_interval):
    trainer.train(save_interval, 64, tensorboard_on=False, save_interval=None, print_on=False)
    trainer.performance_stats(cfg.data_config.labels, save_path=f"{PACKAGE_PATH}/output/Finetuned_SDLCD_2.csv")
    save_net(trainer.net, "Finetuded_SDLCD", model_dir)
    print(i+save_interval)
    

