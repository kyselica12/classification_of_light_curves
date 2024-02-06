import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import glob
import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.data_load import load_data
from src.data.filters import filter_data
from src.nn.datasets.utils import create_datasets, find_dataset_class
from src.config import DataConfig, PACKAGE_PATH, FCConfig, CNNConfig, CNNFCConfig
from src.nn.networks.net import BaseNet
from src.nn.networks.utils import save_net
from src.nn.networks.center_loss import CenterLoss


class Trainer:

    def __init__(self, net, net_config, sampler=False, device='cpu') -> None:

        self.name = net_config.name
        self.save_path = net_config.save_path
        self.model_config = net_config.model_config

        self.central_loss = False
        self.central_loss_weight = 1.

        if isinstance(self.model_config, FCConfig):
           self.feature_dim = self.model_config.layers[-1] 
        elif isinstance(self.model_config, CNNConfig):
            self.feature_dim = self.model_config.classifier_layers[-1]
        elif isinstance(self.model_config, CNNFCConfig):
            self.feature_dim = self.model_config.classifier_layers[-1]

        self.n_classes = net_config.model_config.output_size
        self.net: BaseNet = net
        self.train_set = []
        self.val_set = []
        self.sampler = None
        self.class_weights = None
        
        self.device = device
        self.optim = None
        self.optim_cent = None

    def load_data(self, cfg: DataConfig):
        self.data = load_data(cfg.path, cfg.labels, cfg.regexes, cfg.convert_to_mag)
        if cfg.filter:
            self.data = filter_data(self.data, cfg.filter)

        self.train_set, self.val_set = create_datasets(self.data, cfg)

    def add_sampler(self):
        labels_unique, counts = np.unique(self.train_set.labels, return_counts=True)
        weights = [sum(counts) / c  for c in counts]
        if len(weights) < 5:
            weights.append(1)
        self.class_weights = torch.tensor(weights).to(self.device)
        print(self.class_weights)
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
        train_loader = DataLoader(self.train_set,batch_size=batch_size, shuffle=self.sampler is None, sampler=self.sampler)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(self.train_set,batch_size=batch_size, shuffle=self.sampler is None, sampler=self.sampler)
        val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        self.net.to(self.device)
        self.net.double()
        self.net.train()
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights) 
        criterion_cent = CenterLoss(num_classes=self.n_classes, feat_dim=self.feature_dim,use_gpu=self.device != "cpu")

        if self.optim is None or reset_optimizer:
            self.optim = optim.Adam(self.net.parameters(), lr=0.001)
        if self.central_loss and self.optim_cent is None:
            self.optim_cent = optim.Adam(criterion_cent.parameters(), lr=0.5)

        if tensorboard_on:
            tensorboard = SummaryWriter(log_dir=f"{PACKAGE_PATH}/tensorboard/run")

        val_losses = []
        train_losses = []
        for epoch in tqdm.tqdm(range(epochs), desc="Training", position=0):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0

            for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}: ", leave=False, position=1):
                # get the inputs; data is a list of [inputs, labels]
                epoch_loss, epoch_correct = self._train_one_epoch(criterion, criterion_cent, data)

                # print statistics
                running_loss += epoch_loss
                correct += epoch_correct
                total += data[1].size(0)

            running_loss /= total

            if save_interval is not None and (epoch + 1) % save_interval == 0:
                save_net(self.net, self.save_path, self.name)
                self.net.checkpoint += 1

            val_acc, val_loss, _ = self.evaulate_dataset(val_loader)
            val_losses.append(val_loss)
            train_losses.append(running_loss)
            
            if print_on:
                print(f"Train:\n\tLoss: {running_loss}\n\tAcc: {correct/total*100}", flush=True)
                print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)

            self.net.epoch_trained += 1

            if tensorboard_on:
                tensorboard.add_scalars(f"{self.name}/accuracy", {'train':correct/total * 100,'val':val_acc}, self.net.epoch_trained)
                tensorboard.add_scalars(f"{self.name}/loss", {'train':running_loss,'val': val_loss}, self.net.epoch_trained)
            
        if tensorboard_on:
            tensorboard.close()
        
        return train_losses, val_losses
 
    def _train_one_epoch(self, criterion, criterion_cent, data):
        inputs, labels = data
        
        if torch.any(torch.isnan(inputs)):
            raise Exception("NaN in inputs")
        
        inputs = inputs.to(self.device).double()
        labels = labels.to(self.device)
                
        if torch.any(torch.isnan(inputs)):
            raise Exception("NaN in inputs")

        self.optim.zero_grad()
        loss_cent = 0
        features = None
        outputs = self.net.forward(inputs, features=self.central_loss)
        if self.central_loss:
            self.optim_cent.zero_grad()
            features, outputs = outputs
            loss_cent = criterion_cent(features.float(), labels.float())

        loss_ce = criterion(outputs, labels.long())
        loss = loss_ce + self.central_loss_weight * loss_cent
        loss.backward()
        self.optim.step()

        if self.central_loss:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / self.central_loss_weight)
            self.optim_cent.step()
        
        predicted = torch.argmax(outputs, dim=1).flatten()
        correct = (predicted == labels).sum().item()

        return loss.item(), correct

    def evaulate_dataset(self, data_loader, n_labels=None):
        total = 0
        correct = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(weight=self.class_weights) 

        pred_y = np.empty((0,0))
        true_y = np.empty((0,0))

        with torch.no_grad():   
            for data in data_loader:
                inputs, labels = data
                labels = labels.to(self.device)
                
                inputs = inputs.to(self.device).double()
                outputs = self.net(inputs)

                predicted = torch.argmax(outputs, dim=1).flatten()
                # _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels.long())

                total_loss += loss.item()
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                if self.device != "cpu":
                    predicted = predicted.detach().to('cpu')
                    labels = labels.detach().to('cpu')
                pred_y = np.append(pred_y, predicted.numpy())
                true_y = np.append(true_y, labels.numpy())

        
            labels = None if n_labels is None else list(range(n_labels))
            conf_matrix = confusion_matrix(true_y, pred_y, labels=labels)

            return 100 * correct / total, total_loss/total, conf_matrix

    def performance_stats(self, labels, show=True, save_path=None):
        
        
        train_loader = DataLoader(self.train_set,batch_size=64, sampler=self.sampler is None, shuffle=not self.sampler)
        val_loader = DataLoader(self.val_set, batch_size=64, shuffle=True)
        train_loader = DataLoader(self.train_set,batch_size=64, shuffle=False)
        val_loader = DataLoader(self.val_set, batch_size=64, shuffle=False)


        train_acc, train_loss, _ = self.evaulate_dataset(train_loader, len(labels)) 
        val_acc, val_loss, conf_matx = self.evaulate_dataset(val_loader, len(labels))

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
                name=self.name,
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
