import numpy as np
import random
import torch 
import tqdm
import random
from torch.utils.data import Dataset
from src.configs import AugmentType as A
from src.configs import MIX_UP_ALPHA


class LCDataset(Dataset):
    def __init__(self, data, labels, n_classes, augmentations=[]):
        self.data = data
        self.labels = labels.astype(np.int32)
        self.n_classes = n_classes
        self.augmentations = augmentations

        self.use_cyclic_augmentation = False
        self.use_mixup = False
        self.add_noise = False

        if A.SHIFT in augmentations:
            self.use_cyclic_augmentation = True
        if A.MIX_UP in augmentations:
            self.use_mixup = True
        if A.NOISE in augmentations:
            self.add_noise = True
            # self.align_phases()

        self.data_by_class = {i: self.data[labels==i] for i in range(n_classes)}

    def __len__(self):
        return len(self.data)
    
    def cyclic_augmentation(self, x):
        shift = np.random.randint(0, x.shape[0])
        return np.roll(x, shift, axis=0)
    
    def align_phases(self, shift_step=1):

        for i in tqdm.tqdm(range(1,len(self.data)), desc="Aligning phases"):
            u = self.data[i-1] # prev LC
            v = self.data[i]
            self.data[i] = min([np.roll(v, r, axis=0) for r in range(0, v.shape[-1],shift_step)], 
                     key=lambda vv: np.linalg.norm(vv-u))

    def mixup(self, sample, label):
        
        label2 = random.choice([i for i in range(self.n_classes) if i != label])
        M = self.data_by_class[label2].shape[0]
        idx2 = random.randrange(0,M-1)
        sample2 = self.data_by_class[label2][idx2]


        if MIX_UP_ALPHA > 0:
            lam = np.random.beta(MIX_UP_ALPHA, MIX_UP_ALPHA)
        else:
            lam = 1
        
        mixed_label = np.zeros(self.n_classes)

        mixed_label[label] = lam
        mixed_label[label2] = 1-lam

        mixed_sample = lam * sample + (1 - lam) * sample2

        return mixed_sample, label2, lam
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        out = {"data":sample, "target":label}

        if self.use_mixup:
            mixed_sample, label2, lam = self.mixup(sample, label)
            out["target2"] = label2
            out["lam"] = lam
            out["data"] = mixed_sample
        
        if self.use_cyclic_augmentation:
            out["data"] = self.cyclic_augmentation(out["data"])
        
        if self.add_noise:
            mmax = np.max(out["data"])
            mmin = np.min(out["data"])

            noise = np.random.randn(*out["data"].shape) * (mmax - mmin) * 0.05
            out["data"] += noise
        

        return out
