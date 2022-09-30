import numpy as np
from torch.utils.data import Dataset
import random

from config import (AUGMENTATION_GAP_PROB, AUGMENTATION_LEAVE_ORIGINAL, 
                    AUGMENTATION_MAX_LEN, AUGMENTATION_MAX_NOISE, 
                    AUGMENTATION_MIN_EXAMPLES, AUGMENTATION_MIN_LEN, 
                    AUGMENTATION_NUM_GAPS)

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


class AugmentedBalancedDataset(NetDataset):

    def __init__(self, data, labels) -> None:

        self.max_noise = AUGMENTATION_MAX_NOISE
        
        self.num_gaps = AUGMENTATION_NUM_GAPS
        self.gap_prob = AUGMENTATION_GAP_PROB
        self.min_len = AUGMENTATION_MIN_LEN
        self.max_len = AUGMENTATION_MAX_LEN

        self.data = data
        self.labels = labels

        self._normalize_data()

        self.augment_data(AUGMENTATION_LEAVE_ORIGINAL, AUGMENTATION_MIN_EXAMPLES)

        self._shuffle_data()

        self.data = data
        self.labels = labels

    def augment_data(self, leave_original, class_size):
        unique_labels = np.unique(self.labels)

        augmented_data = []
        augmented_labels = []
        for l in unique_labels:
            
            label_data = self.data[self.labels == l].copy()
            augmented_label_data = []
            if  leave_original:
                augmented_label_data = list(label_data)

            i = 0
            while len(augmented_label_data) < class_size:
                d = self._get_augmented(label_data[d])
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
        
    def _get_augmented(self, data):
        return self._add_gap(self._roll(self._add_noise(data)))
    
    def _add_noise(self, data):

        noise = np.random.randn(*data.shape) * self.max_noise
        return data + noise
    
    def _roll(self, data):
        
        shift = random.randrange(max(data.shape))

        return np.roll(data, shift)

    def _add_gap(self, data):
        data = data.copy()
        DATA_SHAPE = max(data.shape)
        for _ in range(self.num_gaps):
            if random.random() <= self.gap_prob:
                size = random.randrange(self.min_len, self.max_len + 1)
                index = random.randrange(max(data.shape))
                data[np.linspace(index, index + size - 1, num=size).astype(np.int32) - DATA_SHAPE] = 0

        return data



def create_datasets(labeled_data, labels, validation_split=0.1, output_folder=None):
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

    idx_train, idx_test = np.random.permutation(len(X_train)), np.random.permutation(len(X_test))

    X_train, X_test = X_train[idx_train], X_test[idx_test]
    Y_train, Y_test = Y_train[idx_train], Y_test[idx_test]


    for label, idx in labels_id.items():
        print(f"label: {label} -> {np.sum(Y_train == idx)} training examples, {np.sum(Y_test == idx)} testing examples")

    val_set = NetDataset(X_test, Y_test)
    train_set = NetDataset(X_train, Y_train)

    if output_folder:
        np.savetxt(f"{output_folder}/train_x.np", X_train)
        np.savetxt(f"{output_folder}/train_y.np", Y_train)
        np.savetxt(f"{output_folder}/test_x.np", X_test)
        np.savetxt(f"{output_folder}/test_y.np", Y_test)

    return train_set, val_set