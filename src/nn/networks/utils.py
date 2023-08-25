import numpy as np
import torch
import random
from pydoc import locate
import glob
import os

from src.config import NetConfig, Config


def get_new_net(cfg: Config, save_config_path=None):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cfg.seed = seed

    print(f"SEED: {seed}")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.net_config.name = f"{cfg.net_config.name}_{seed}"
    NetClass = locate_net_class(cfg.net_config.net_class)
    net = NetClass(cfg.net_config.model_config)

    if save_config_path is not None:
        with open(save_config_path, "w") as f:
            print(cfg.to_json(), file=f)

    return net

def get_checkpoint_and_epoch_number(path):
    checkpoint = int(os.path.split(path)[1][-9:-6])
    epoch = int(path.split("_")[-3])
    seed = int(path.split("_")[-5])
    return checkpoint, epoch, seed

def load_net(cfg: NetConfig, seed, checkpoint="latest"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cfg.seed = seed
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.net_config.name = f"{cfg.net_config.name}_{seed}"
    # net = ResNet(net_cfg.n_classes, device=device, name=net_cfg.name)
    NetClass = locate_net_class(cfg.net_config.net_class)
    net = NetClass(cfg.net_config.model_config)

    if checkpoint == 'latest':
        models = list(glob.iglob(f"{cfg.save_path}\\{cfg.name}_epochs_*_checkpoint_*.model"))
        model_path = max(models, key=get_checkpoint_and_epoch_number)
    else:
        models = list(glob.iglob(f"{cfg.save_path}\\{cfg.name}_epochs_*_checkpoint_{checkpoint:03d}.model"))
        model_path = max(models, key=get_checkpoint_and_epoch_number)

    net.load_state_dict(torch.load(model_path))
    net.checkpoint, net.epoch_trained, seed = get_checkpoint_and_epoch_number(model_path)
    print(f"SEED: {seed}")
    return net

def save_net(net, name, save_path):
    torch.save(net.state_dict(), f'{save_path}\\{name}_epochs_{net.epoch_trained:d}_checkpoint_{net.checkpoint:03d}.model')

def locate_net_class(name):
    class_path = f"src.nn.networks.{name.lower()}.{name}"
    NetClass = locate(class_path)
    return NetClass