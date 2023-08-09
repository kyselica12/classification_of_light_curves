import numpy as np
import torch
import random
from pydoc import locate

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
    net = NetClass(cfg.net_config)

    if save_config_path is not None:
        with open(save_config_path, "w") as f:
            print(cfg.to_json(), file=f)

    return net

def load_net(cfg: NetConfig, seed, checkpoint="latest"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cfg.seed = seed
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.net_config.name = f"{cfg.net_config.name}_{seed}"
    # net = ResNet(net_cfg.n_classes, device=device, name=net_cfg.name)
    NetClass = locate_net_class(cfg.net_config.net_class)
    net = NetClass(cfg.net_config)
    net.load(checkpoint)

    return net

def locate_net_class(name):
    class_path = f"src.nn.{name.lower()}.{name}"
    print(class_path)
    NetClass = locate(class_path)
    return NetClass