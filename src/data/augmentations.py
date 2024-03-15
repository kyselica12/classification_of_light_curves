from typing import List
import numpy as np
from abc import ABC, abstractmethod

import torch


class Augmentation(ABC):
    @abstractmethod
    def apply(self, x):
        pass

class CyclicShift(Augmentation):
    def apply(self, x):
        shift = np.random.randint(0, x.shape[0])
        return np.roll(x, shift, axis=0)

def compose(augmentations: List[Augmentation]):
    def f(x):
        for aug in augmentations:
            x = aug.apply(x)
        return x
    return f
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    # Align the phases of LCs -> lowest distance between the two LCs
    mixed_x = x.clone()
    shift_step = 10
    for i, j in enumerate(index):
        v,u = x[i],x[j]
        vs = min([torch.roll(v, r, dims=0) for r in range(0, v.shape[-1],shift_step)], 
                 key=lambda vv: torch.norm(vv-u))
        mixed_x[i] = lam * vs + (1 - lam) * u

    # mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    