import numpy as np
from torch.utils.data import Dataset
import random
from scipy import optimize
from pydoc import locate

from src.config import DataConfig

class NetDataset(Dataset):

    def __init__(self, data, labels) -> None:
        self.data = data
        self.labels = labels

        NetDataset._normalize_data(self.data)
    
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

class FourierDataset(Dataset):

    def __init__(self, data, labels, std, residuals, rms, amplitude) -> None:
        
        self.use_std = std
        self.use_residuals = residuals
        self.use_rms = rms
        self.use_amplitude = amplitude
        
        self.data = np.array(list(map(self._preprocess_example, data)))
        self.labels = labels


    
    def _fourier8(self, x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
        pi = np.pi
        y = a0 + a1 * np.cos(x * 2*pi) + b1 * np.sin(x * 2*pi) + \
            a2 * np.cos(2 * x * 2*pi) + b2 * np.sin(2 * x * 2*pi)+ \
            a3 * np.cos(3 * x * 2*pi) + b3 * np.sin(3 * x * 2*pi) + \
            a4 * np.cos(4 * x * 2*pi) + b4 * np.sin(4 * x * 2*pi) + \
            a5 * np.cos(5 * x * 2*pi) + b5 * np.sin(5 * x * 2*pi) + \
            a6 * np.cos(6 * x * 2*pi) + b6 * np.sin(6 * x * 2*pi) + \
            a7 * np.cos(7 * x * 2*pi) + b7 * np.sin(7 * x * 2*pi) + \
            a8 * np.cos(8 * x * 2*pi) + b8 * np.sin(8 * x * 2*pi) 
        return y

    def _push_to_max(self, example):

        example[example == 0] = np.inf

        minimal_y_value = np.amin(example)
        index_minimal = np.where(example == minimal_y_value)[0][0]
        
        example = np.roll(example, -index_minimal)
        
        example[example == np.inf] = 0
        
        return example

    def _foufit(self, example):
        example = self._push_to_max(example)
        phases = np.linspace(0, 1, len(example), endpoint=False)

        non_zero = example != 0
        xs = phases[non_zero]
        ys = example[non_zero]

        params, params_covariance = optimize.curve_fit(self._fourier8, xs, ys, absolute_sigma=False, method="lm", maxfev=10000)
        std = np.sqrt(np.diag(params_covariance))

        y_hat = self._fourier8(phases, *params)
        
        amplitude = np.max(y_hat) - np.min(y_hat)
        
        residuals = np.abs(example - y_hat) / (amplitude + 1e-6)
        residuals[np.logical_not(non_zero)] = 0
        
        rms = np.sqrt(np.sum(residuals[non_zero]**2) / (residuals[non_zero].size-2))

        return params, std, residuals, rms, amplitude

    def _preprocess_example(self, example):
        params, std, residuals, rms, amplitude = self._foufit(example)

        res = params
        
        if self.use_std:
            res = np.concatenate((res, std))
        
        if self.use_residuals:
            res = np.concatenate((res, residuals))

        if  self.use_rms:
            res = np.concatenate((res, [rms]))
        
        if  self.use_amplitude:
            res = np.concatenate((res, [amplitude]))

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        arr = self.data[index]
        label = self.labels[index]

        return arr, label


def split_data_to_test_validation(labeled_data, labels, max_number_of_training_examples=None, validation_split=0.1):
    X_test, X_train = [], []
    Y_test, Y_train = [], []

    labels_id = {l: i for i, l in enumerate(labels)}

    for obj in labeled_data:
        x = labeled_data[obj]
        N = int(len(x))
        y = [labels_id[obj]]*N
        
        if max_number_of_training_examples is None:
            max_number_of_training_examples = N
        
        n = int(N * (1 - validation_split))
        n = min(n, max_number_of_training_examples)

        random.shuffle(x)

        X_test.extend(x[n:])
        Y_test.extend(y[n:])

        X_train.extend(x[:n])
        Y_train.extend(y[:n])


    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = np.array(Y_train, dtype=np.int32), np.array(Y_test, dtype=np.int32)

    return (X_train, Y_train), (X_test, Y_test)

def create_datasets(labeled_data, cfg:DataConfig):

    (X_train, Y_train), (X_test, Y_test) = split_data_to_test_validation(labeled_data, cfg.labels, cfg.number_of_training_examples_per_class, cfg.validation_split)

    idx_train, idx_test = np.random.permutation(len(X_train)), np.random.permutation(len(X_test))

    X_train, X_test = X_train[idx_train], X_test[idx_test]
    Y_train, Y_test = Y_train[idx_train], Y_test[idx_test]

    DatasetClass = locate(f'src.nn.dataset.{cfg.dataset_class}')
    val_set = DatasetClass(X_test, Y_test, **cfg.dataset_arguments)
    train_set = DatasetClass(X_train, Y_train, **cfg.dataset_arguments)

    print(f"Training set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")

    if cfg.save_path:
        np.save(f"{cfg.save_path}/train_x.np", X_train)
        np.save(f"{cfg.save_path}/train_y.np", Y_train)
        np.save(f"{cfg.save_path}/test_x.np", X_test)
        np.save(f"{cfg.save_path}/test_y.np", Y_test)

    return train_set, val_set

if __name__ == "__main__":
    class_name = "FourierDataset"
    c = locate(f'nn.dataset.{class_name}')
    c2 = locate(f'src.nn.dataset.{class_name}')
    
    print(c, c2)