from config import DataConfig, NetConfig, FilterConfig
from data.data_load import load_data
from data.filters import filter_data
from nn.net import Net, evaulate_net
from nn.dataset import create_datasets
import numpy as np
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

class Trainer:

    def __init__(self, net, data_cfg: DataConfig, filter_cfg: FilterConfig, net_cfg: NetConfig) -> None:
        
        self.data = load_data(data_cfg)
        filtered_data = filter_data(self.data, filter_cfg)

        self.net = net
        
        self.train_set, self.val_set = create_datasets(filtered_data, data_cfg.labels, validation_split=0.1)
        # labels_unique, counts = np.unique(self.train_set.labels, return_counts=True)
        # self.class_weights = [sum(counts) / c  for c in counts]
        # example_weights = [self.class_weights[e] for e in self.train_set.labels ]
        # sampler = WeightedRandomSampler(example_weights, len(self.train_set.labels))

        self.train_loader = DataLoader(self.train_set,batch_size=64)
        self.val_loader = DataLoader(self.val_set, batch_size=64)
    
    def train(self, epochs: int, reset_optimizer=False, tensorboard_on=False, print_on=False, save_interval=None) -> None:
        self.net.train_model(self.train_loader, self.val_loader, epochs, 
                             reset_optimizer,tensorboard_on, save_interval, print_on, class_weights=None)

    def evaluate(self, labels=None):
        if labels:
            for i in range(len(labels)):
                print(i, labels[i])

        train_acc, train_loss, _ = evaulate_net(self.net, self.train_loader)
        val_acc, val_loss, conf_matx = evaulate_net(self.net, self.val_loader)

        print(f"Train:\n\tLoss: {train_loss}\n\tAcc: {train_acc}", flush=True)
        print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)

        print(conf_matx)

        precision = []
        recall = []
        for i in range(len(conf_matx)):
            p = conf_matx[i][i] / np.sum(conf_matx[i]) * 100
            precision.append(p)

            r = conf_matx[i][i] / np.sum(conf_matx[:, i]) * 100
            recall.append(r)

        print("Precision:\n", precision)
        print("Recall:\n", recall)

    