import sys

sys.path.append("./")

from src.config import  PACKAGE_PATH, FourierDatasetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *
from src.experiments.utils import get_default_cfg, load_dataset_to_trainer
from src.main import Trainer
from src.nn.networks.utils import get_new_net, load_net, save_net
from src.experiments.utils import test_SDLCD
from src.nn.networks.fc import FC

import torch
import numpy as np  


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

SDLCD_data_path = f"{PACKAGE_PATH}/output/datasets/SDLCD"
test_set = FourierDataset([],[],**cfg.data_config.dataset_arguments)
test_set.data = np.load(f"{SDLCD_data_path}/test_x.npy")
test_set.labels = np.load(f"{SDLCD_data_path}/test_y.npy").astype(dtype=np.int32)
test_set.offset = 16    
test_set.compute_std_mean()
mean, std = test_set.mean, test_set.std

import json

LI_JSON_PATH = f"{PACKAGE_PATH}/resources/SDLCD/DATA_LI.json"

with open(LI_JSON_PATH) as json_file:
    data = json.load(json_file)

X = {}
for d in data:
    for _, v in d.items():
        array = [v[c+str(i)] for c in "ab" for i in range(1,9)]
        # array = (np.array(array) - mean) / std
        array = torch.tensor(array, dtype=torch.float64)
        X[v["Points"]] = array

net.eval()
for k in X:
    output = net(X[k].unsqueeze(0))
    output = torch.softmax(output, dim=1)
    output = output.detach().numpy() * 100
    # pretty print the results
    classes = cfg.data_config.labels
    output = [f"{classes[i]}: {output[0][i]:.2f}%" for i in range(5)]
    print(k, output)



