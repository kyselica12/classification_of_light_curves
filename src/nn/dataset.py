import numpy as np
from torch.utils.data import Dataset
import random

from config import DataConfig, AugmentationConfig

class NetDataset(Dataset):

    def __init__(self, data, labels) -> None:
        self.data = data
        self.labels = labels

        self._normalize_data()

    def _normalize_data(self):
        max_value = np.array([np.max(d[d!=0]) for d in self.data])
        min_value = np.array([np.min(d[d!=0]) for d in self.data])

        print(min_value)
        for i in range(len(self.data)):
            d = self.data[i]
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

class AugmentedBalancedDataset(NetDataset):

    def __init__(self, 
                    data, labels, 
                    roll=True, add_gaps=True, add_noise=True,
                    max_noise=None, 
                    num_gaps=None, min_gap_len=None, max_gap_len=None, gap_prob=None, 
                    use_original_data=True, min_num_examples=1000) -> None:

        self.max_noise = max_noise
        
        self.num_gaps = num_gaps
        self.gap_prob = gap_prob
        self.min_len = min_gap_len
        self.max_len = max_gap_len

        self.data = data
        self.labels = labels


        self._normalize_data()

        self.augment_data(use_original_data, min_num_examples, roll, add_gaps, add_noise)

        self._shuffle_data()


    def augment_data(self, leave_original, class_size, roll, add_gaps, add_noise):
        unique_labels = np.unique(self.labels)

        augmented_data = []
        augmented_labels = []
        for l in unique_labels:
            
            label_data = self.data[self.labels == l].copy()
            augmented_label_data = []
            if  leave_original:
                augmented_label_data = list(label_data)

            if roll or add_gaps or add_noise:
                i = 0
                while len(augmented_label_data) < class_size:
                    d = self._get_augmented(label_data[i], roll, add_gaps, add_noise)
                    augmented_label_data.append(d)
                    i = (i + 1) % len(label_data)

            augmented_data.extend(augmented_label_data)
            augmented_labels.extend([l]*len(augmented_label_data))
        
        self.data = np.array(augmented_data)
        self.labels = np.array(augmented_labels).astype(np.int32)
     
    def _shuffle_data(self):
        shuffle_indices = np.random.permutation(len(self.data))

        self.data = self.data[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        
    def _get_augmented(self, data, roll, add_gaps, add_noise):
        if add_noise:
            data = self._add_noise(data)
        if roll:
            data = self._roll(data)
        if add_gaps:
            data = self._add_gap(data)
        
        return data
    
    def _add_noise(self, data):

        noise = np.random.randn(*data.shape) * self.max_noise
        return data + noise
    
    def _roll(self, data):
        
        shift = random.randrange(max(data.shape))

        return np.roll(data, shift)
        # return data

    def _add_gap(self, data):
        data = data.copy()
        DATA_SHAPE = max(data.shape)
        for _ in range(self.num_gaps):
            if random.random() <= self.gap_prob:
                size = random.randrange(self.min_len, self.max_len + 1)
                index = random.randrange(max(data.shape))
                data[np.linspace(index, index + size - 1, num=size).astype(np.int32) - DATA_SHAPE] = 0

        return data

def create_dataset(data, labels, cfg: AugmentationConfig):

    if cfg:
        return AugmentedBalancedDataset(data, labels,
                                        roll=cfg.roll, 
                                        add_gaps=cfg.add_gaps,
                                        add_noise=cfg.add_noise,
                                        max_noise=cfg.max_noise,
                                        num_gaps=cfg.num_gaps,
                                        min_gap_len=cfg.min_gap_len,
                                        max_gap_len=cfg.max_gap_len,
                                        gap_prob=cfg.gap_prob,
                                        use_original_data=cfg.keep_original,
                                        min_num_examples=cfg.min_examples
        )

    return NetDataset(data, labels)

def split_data(labeled_data, labels, validation_split=0.1):
    X_test, X_train = [], []
    Y_test, Y_train = [], []

    labels_id = {l: i for i, l in enumerate(labels)}

    for obj in labeled_data:
        x = labeled_data[obj]
        N = int(len(x))
        y = [labels_id[obj]]*N
        
        n = int(N * validation_split)
        random.shuffle(x)

        X_test.extend(x[:n])
        Y_test.extend(y[:n])

        X_train.extend(x[n:])
        Y_train.extend(y[n:])


    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = np.array(Y_train, dtype=np.int32), np.array(Y_test, dtype=np.int32)

    return (X_train, Y_train), (X_test, Y_test)

def create_datasets(labeled_data, cfg:DataConfig):

    (X_train, Y_train), (X_test, Y_test) = split_data(labeled_data, cfg.labels, cfg.validation_split)

    idx_train, idx_test = np.random.permutation(len(X_train)), np.random.permutation(len(X_test))

    X_train, X_test = X_train[idx_train], X_test[idx_test]
    Y_train, Y_test = Y_train[idx_train], Y_test[idx_test]

    val_set = create_dataset(X_test, Y_test, cfg.augmentation)
    train_set = create_dataset(X_train, Y_train, cfg.augmentation)

    print(f"Training set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")

    if cfg.save_path:
        np.save(f"{cfg.save_path}/train_x.np", X_train)
        np.save(f"{cfg.save_path}/train_y.np", Y_train)
        np.save(f"{cfg.save_path}/test_x.np", X_test)
        np.save(f"{cfg.save_path}/test_y.np", Y_test)

    return train_set, val_set