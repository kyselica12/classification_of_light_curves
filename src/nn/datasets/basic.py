import numpy as np
from torch.utils.data import Dataset

class BasicDataset(Dataset):

    def __init__(self, data, labels) -> None:
        self.data = data
        self.labels = labels

        BasicDataset._normalize_data(self.data)
    
    @staticmethod
    def _normalize_data(data):
        max_value = np.array([np.max(d[d!=0]) for d in data])
        min_value = np.array([np.min(d[d!=0]) for d in data])

        for i in range(len(data)):
            d = data[i]
            if max_value[i] != 0:
                d[d==0] = -1
                diff = max_value[i] - min_value[i]
                d[d!=-1] = (d[d!=-1] - min_value[i]) / (diff + 0.000000000001)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arr = self.data[index]
        label = self.labels[index]

        return arr, label