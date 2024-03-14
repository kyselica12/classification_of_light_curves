import numpy as np
import random
import torch 
from torch.utils.data import Dataset


class LCDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def cyclic_augmentation(self, x):
        shift = np.random.randint(0, x.shape[0])
        return np.roll(x, shift, axis=0)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label