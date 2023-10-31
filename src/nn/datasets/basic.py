import numpy as np
from torch.utils.data import Dataset

class BasicDataset(Dataset):

    def __init__(self, data, labels, mode='val') -> None:
        self.labels = labels
        self.data = self._normalize_data(data)
    
    def _normalize_data(self, data):
        max_value = np.array([np.max(d[d!=0]) for d in data])
        min_value = np.array([np.min(d[d!=0]) for d in data])
        res = []
        for i in range(len(data)):
            d = data[i]
            if max_value[i] != 0:
                diff = max_value[i] - min_value[i]
                d[d!=0] = (d[d!=0] - min_value[i] + 1e-5) / (diff + 1e-5)
                # d[d==0] = -1
            res.append(d)
        return np.array(res).astype(np.double)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arr = self.data[index]
        label = self.labels[index]

        return arr, label
