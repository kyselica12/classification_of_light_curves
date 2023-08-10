import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.data_load import load_data
from src.data.filters import filter_data
from src.nn.datasets.basic import BasicDataset
from src.nn.datasets.utils import create_datasets, find_dataset_class
from src.config import DataConfig
from src.nn.networks.net import BaseNet


class Trainer:

    def __init__(self, net, sampler=False) -> None:

        self.net: BaseNet = net
        self.train_set = []
        self.val_set = []
        self.sampler = None
        self.class_weights = None
        self.train_loader = None
        self.val_loader = None

    def load_data(self, cfg: DataConfig):
        self.data = load_data(cfg.path, cfg.labels, cfg.regexes, cfg.convert_to_mag)
        if cfg.filter:
            self.data = filter_data(self.data, cfg.filter)

        self.train_set, self.val_set = create_datasets(self.data, cfg)

    def add_sampler(self):
        labels_unique, counts = np.unique(self.train_set.labels, return_counts=True)
        self.class_weights = torch.tensor([sum(counts) / c  for c in counts]).to(self.net.device)
        example_weights = [self.class_weights[e] for e in self.train_set.labels ]
        self.sampler = WeightedRandomSampler(example_weights, len(self.train_set.labels))

    def save_data(self, path):
        np.save(f"{path}/train_x.npy", self.train_set.data)
        np.save(f"{path}/train_y.npy", self.train_set.labels)
        np.save(f"{path}/val_x.npy", self.val_set.data)
        np.save(f"{path}/val_y.npy", self.val_set.labels)

    def load_data_from_file(self, path, cfg: DataConfig):
        DatasetClass = find_dataset_class(cfg.dataset_class)

        self.train_set = DatasetClass([],[],**cfg.dataset_arguments)
        self.train_set.data = np.load(f"{path}/train_x.npy")
        self.train_set.labels = np.load(f"{path}/train_y.npy").astype(dtype=np.int32)
        
        self.val_set = DatasetClass([],[],**cfg.dataset_arguments)
        self.val_set.data = np.load(f"{path}/val_x.npy")
        self.val_set.labels = np.load(f"{path}/val_y.npy").astype(dtype=np.int32)

    
    def train(self, epochs: int, batch_size:int, reset_optimizer=False, tensorboard_on=False, print_on=False, save_interval=None) -> None:
        self.train_loader = DataLoader(self.train_set,batch_size=batch_size, sampler=self.sampler)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size)
        self.net.train_model(self.train_loader, self.val_loader, epochs, 
                             reset_optimizer,tensorboard_on, save_interval, print_on, class_weights=self.class_weights)

    def evaulate(self, labels, show=True, save_path=None):
        
        
        self.train_loader = DataLoader(self.train_set,batch_size=64, sampler=self.sampler)
        self.val_loader = DataLoader(self.val_set, batch_size=64)


        train_acc, train_loss, _ = self.net.evaulate(self.train_loader) 
        val_acc, val_loss, conf_matx = self.net.evaulate(self.val_loader)
        confusion_matrix_data = [[labels[i]] + list(conf_matx[i])  for i in range(len(labels))]
        df_confusion_matrix = pd.DataFrame(confusion_matrix_data, columns=["Label"] + labels)

        precision = []
        recall = []
        for i in range(len(conf_matx)):
            p = conf_matx[i][i] / np.sum(conf_matx[i]) * 100
            precision.append(p)

            r = conf_matx[i][i] / np.sum(conf_matx[:, i]) * 100
            recall.append(r)

        f1_score = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))
        
        df_precision = pd.DataFrame([precision, recall, list(f1_score)], ["Precision", "Recall", "F1 score"], labels)



        if show:
            print(f"Train:\n\tLoss: {train_loss}\n\tAcc: {train_acc}", flush=True)
            print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)
            print("-----------------------------------------\n")
            print(df_confusion_matrix)
            print("\n-----------------------------------------\n")
            print(df_precision)
            print("\n-----------------------------------------\n")

        if save_path:
            
            df_out = df_confusion_matrix.copy()
            df_out = df_out.assign(
                name=self.net.name,
                epochs=self.net.epoch_trained,
                net=str(self.net),
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                precision=precision,
                recall=recall,
                f1_score=list(f1_score)
                )

            df_out.to_csv(save_path, mode='a', header=not os.path.exists(save_path))