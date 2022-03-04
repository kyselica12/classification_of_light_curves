import numpy as np
from torch.utils.data import Dataset

class NetDataset(Dataset):

    def __init__(self, data, labels) -> None:
        self.data = data
        self.labels = labels

        self._normalize_data()

    def _normalize_data(self):
        max_value = np.array([np.max(d[d!=0]) for d in self.data])
        min_value = np.array([np.min(d[d!=0]) for d in self.data])

        for i in range(len(self.data)):
            d = self.data[i]
            if max_value[i] != 0:
                d[d==0] = -1
                diff = max_value[i] - min_value[i]
                d[d!=-1] = (d[d!=-1] - min_value[i]) / diff

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arr = self.data[index]
        label = self.labels[index]

        return arr, label

    

def create_datasets(labeled_data, labels, validation_split=0.1):
    X = []
    y = []
    
    labels_id = {l: i for i, l in enumerate(labels)}

    for obj in labeled_data:
        arr = labeled_data[obj]
        n = int(len(arr))

        X.extend(arr)
        y.extend([labels_id[obj]]*n)

    X = np.array(X)
    y = np.array(y)

    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    c_train = int(len(X) * validation_split)

    val_set = NetDataset(X[:c_train], y[:c_train])
    train_set = NetDataset(X[c_train:], y[c_train:])

    return train_set, val_set