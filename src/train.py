from data.data_load import load_data
from data.filters import filter_data
from nn.net import Net, evaulate_net
from nn.dataset import create_datasets

import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

from nn.dataset import NetDataset, AugmentedBalancedDataset
from config import DataConfig


class Trainer:

    def __init__(self, net, sampler=False) -> None:

        self.net = net
        self.train_set = []
        self.val_set = []
        self.sampler = None
        self.class_weights = None
        self.train_loader = None
        self.val_loader = None

    def load_data(self, cfg: DataConfig):
        self.data = load_data(cfg.path, cfg.labels, cfg.convert_to_mag)
        if cfg.filter:
            self.data = filter_data(self.data, cfg.filter)

        self.train_set, self.val_set = create_datasets(self.data, cfg)

    def add_sampler(self):
        labels_unique, counts = np.unique(self.train_set.labels, return_counts=True)
        self.class_weights = torch.tensor([sum(counts) / c  for c in counts]).to(self.net.device)
        example_weights = [self.class_weights[e] for e in self.train_set.labels ]
        self.sampler = WeightedRandomSampler(example_weights, len(self.train_set.labels))

    def load_data_from_file(self, path):
        X_train = np.loadtxt(f"{path}/train_x.npy")
        Y_train = np.loadtxt(f"{path}/train_y.npy").astype(dtype=np.int32)
        X_test = np.loadtxt(f"{path}/test_x.npy")
        Y_test = np.loadtxt(f"{path}/test_y.npy")

        self.val_set = NetDataset(X_test, Y_test)
        self.train_set = NetDataset(X_train, Y_train)     
    
    def train(self, epochs: int, batch_size:int, reset_optimizer=False, tensorboard_on=False, print_on=False, save_interval=None) -> None:
        self.train_loader = DataLoader(self.train_set,batch_size=batch_size, sampler=self.sampler)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size)
        self.net.train_model(self.train_loader, self.val_loader, epochs, 
                             reset_optimizer,tensorboard_on, save_interval, print_on, class_weights=self.class_weights)

    def evaluate(self, labels=None):
        
        if self.train_loader is None:
            self.train_loader = DataLoader(self.train_set,batch_size=64, sampler=self.sampler)
            self.val_loader = DataLoader(self.val_set, batch_size=64)

        if labels:
            for i in range(len(labels)):
                print(i, labels[i])

        train_acc, train_loss, _ = evaulate_net(self.net, self.train_loader)
        val_acc, val_loss, conf_matx = evaulate_net(self.net, self.val_loader)

        print(f"Train:\n\tLoss: {train_loss}\n\tAcc: {train_acc}", flush=True)
        print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)
        print("-----------------------------------------\n")
        df_data = [[labels[i]] + list(conf_matx[i])  for i in range(len(labels))]
        df = pd.DataFrame(df_data, columns=["Label"] + labels)
        print(df)
        print("\n-----------------------------------------\n")

        precision = []
        recall = []
        for i in range(len(conf_matx)):
            p = conf_matx[i][i] / np.sum(conf_matx[i]) * 100
            precision.append(p)

            r = conf_matx[i][i] / np.sum(conf_matx[:, i]) * 100
            recall.append(r)

        df_precision = pd.DataFrame([precision, recall], ["Precision", "Recall"], labels)

        print(df_precision)

        print("\n-----------------------------------------\n")