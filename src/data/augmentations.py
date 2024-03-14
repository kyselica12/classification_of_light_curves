from typing import List
import numpy as np
from abc import ABC, abstractmethod


class Augmentation(ABC):
    @abstractmethod
    def apply(self, x):
        pass

class CyclicShift(Augmentation):
    
    def apply(self, x):
        shift = np.random.randint(0, x.shape[0])
        return np.roll(x, shift, axis=0)

class AugmentationList:
    def __init__(self, augmentations: List[Augmentation]):
        self.augmentations = augmentations
    
    def get_function(self):
        def f(x):
            for aug in self.augmentations:
                x = aug.apply(x)
            return x
        return f

    
    