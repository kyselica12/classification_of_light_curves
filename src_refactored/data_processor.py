import os
import random
import hashlib
import re
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import optimize
import tqdm

from src_refactored.configs import DataConfig, LC_SIZE, FOURIER_N
from src_refactored.filters import filter_data
from src_refactored.dataset import LCDataset


class DataProcessor:
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.class_names = data_config.class_names
        self.regexes = data_config.regexes
        self.path = data_config.path
        self.output_path = data_config.output_path
        self.validation_split = data_config.validation_split
        self.number_of_training_examples_per_class= data_config.number_of_training_examples_per_class
        self.convert_to_mag = data_config.convert_to_mag
        self.max_amplitude = data_config.max_amplitude
        self.filter_config = data_config.filter_config

        self.use_fourier = data_config.fourier
        self.use_std = data_config.std
        self.use_residuals = data_config.residuals
        self.use_rms = data_config.rms
        self.use_amplitude = data_config.amplitude
        self.use_lc = data_config.lc
        self.use_reconstructed = data_config.reconstructed_lc
        self.push_to_max = data_config.push_to_max

        self.examples = None
        self.labels = None
        # ObjectID, TrackID, Phase
        self.headers = None

    def _generate_hash(self):
        hash_text = f'{self.class_names}_{self.regexes}_{self.convert_to_mag}_{self.filter_config}'
        return hashlib.md5(hash_text.encode()).hexdigest()

    def save_data_MMT(self):
        hash = self._generate_hash()
        path = f'{self.output_path}/{hash}'
        os.makedirs(path, exist_ok=True)

        np.savetxt(f"{path}/examples.txt", self.examples)
        np.savetxt(f"{path}/labels.txt", self.labels)
        np.savetxt(f"{path}/headers.txt", self.headers)

    def load_data_MMT(self):
        hash = self._generate_hash()
        path = f'{self.output_path}/{hash}'

        self.examples = np.loadtxt(f'{path}/examples.txt')
        self.labels = np.loadtxt(f'{path}/labels.txt')
        self.headers = np.loadtxt(f'{path}/labels.txt')

    def load_raw_data_MMT(self):
        data_dict, header_dict, columns = self._read_csv_files()

        if self.convert_to_mag:
            self._convert_to_magnitude_in(data_dict)

        if self.filter_config:
            data_dict, header_dict = filter_data(data_dict, header_dict, self.filter_config)

        columns += [f"Fourier coef " for i in range(16)]
        print("Computing Fourier....")

        fourier_coefs = []
        for label, data in data_dict.items():
            for d in tqdm.tqdm(data, desc=f"Fourier for class {label}"):
                fourier_coefs.append(np.concatenate(self._foufit(d)))

    
        self.labels = np.array([i for i, l in enumerate(self.class_names) for _ in range(len(data_dict[l]))])
        self.examples = np.concatenate([data_dict[l] for l in self.class_names])
        self.headers = np.concatenate([header_dict[l] for l in self.class_names])
        # self.examples = np.concatenate(list(data_dict.values()))
        # self.headers = np.concatenate(list(header_dict.values()))
        
        self.examples = np.concatenate((self.examples, np.array(fourier_coefs)), axis=1)

    def prepare_dataset(self, examples, labels, split_strategy="all", seed=None):

        N = len(examples)
        data = []
        lc = examples[:,:LC_SIZE]
        fc_std = examples[:,LC_SIZE:]
        fc = fc_std[:,:2*FOURIER_N+1]
        std = fc_std[:,2*FOURIER_N+1:]

        print(list(fc[0]))

        phases = np.linspace(0, 1, LC_SIZE, endpoint=False)
        print(fc.shape, std.shape, lc.shape)
        y_hat = np.array([self._fourier8(phases, *(list(fc[i]))) for i in range(N)])
        print(y_hat.shape)
        
        amplitude = (np.max(y_hat, axis=1) - np.min(y_hat, axis=1)).reshape(-1,1)
        residuals = np.abs(lc - y_hat) / (amplitude + 1e-6)
        
        if self.use_lc:
            lc[lc == 0] = np.nan
            lc = (lc - np.nanmin(lc, axis=1, keepdims=True) + 1e-6) / (amplitude + 1e-6)
            lc[np.isnan(lc)] = 0
            data.append(lc)
        if self.use_reconstructed:
            recontructed_lc = (y_hat - np.min(y_hat,axis=1, keepdims=True) + 1e-6) / (amplitude + 1e-6)
            data.append(recontructed_lc)
        if self.use_residuals:
            data.append(residuals)
        if self.use_fourier:
            data.append(fc[:,1:])
        if self.use_std:
            data.append(std[:,1:])
        if self.use_rms:
            rms = np.sqrt(np.sum(residuals**2,axis=1) / (np.sum(residuals != 0, axis=1)-2 + 1e-6))
            data.append(rms.reshape(-1,1))
        if self.use_amplitude:
            data.append(amplitude / self.max_amplitude)

        X = np.concatenate(tuple(data), axis=1)
        y = np.array(labels)

        match split_strategy:
            case "all":
                split = self.split_data(X,y,self.validation_split, seed)
            case "objectID" | "trackID":
                header_idx = 0 if split_strategy == "objectID" else 1
                split = self.split_data_by_object(X, y, self.headers, 
                                                   self.number_of_training_examples_per_class,
                                                   self.validation_split,
                                                   split_on_header_idx=header_idx)
            case _:
                raise ValueError(f"Split strategy {split_strategy} not recognized")
        
        (train_X, train_y),(val_X, val_y) = split

        return (train_X, train_y), (val_X, val_y)

    def split_data(self, X,y, split=0.1, seed=None):
        if seed:
            random.seed(seed)
        
        N = X.shape[0]
        indices = list(range(N))
        random.shuffle(indices)

        S = round(N*(1-split))
        train_X = X[:S]
        val_X = X[S:]
        train_y = y[:S]
        val_y = y[S:]

        return (train_X, train_y), (val_X, val_y)

    def split_data_by_object(self, X, Y, headers, k, split=0.1, split_on_header_idx=0):
        for l in range(len(self.class_names)):
            mask = Y == l
            x = X[mask]
            h = headers[mask][:, split_on_header_idx]

            x_obj = [x[h==idx] for idx in np.unique(h)]
            sizes = list(map(len, x_obj))
            N = sum(sizes)

            indices = np.argsort(-np.array(sizes))

            total = 0
            train = np.empty((0, *x[0].shape[1:]))
            val = np.empty((0, *x[0].shape[1:]))

            for i in range(len(indices)):
                if (sizes[indices[i]] + total < k*1.1 and sizes[indices[i]] + total < N * (1-split)) or \
                    (total == 0 and sizes[indices[i]] + total < N * (1-split)):
                    total += sizes[indices[i]]
                    train = np.concatenate((train, x_obj[indices[i]]))
                else:
                    val = np.concatenate((val, x_obj[indices[i]]))

            return train, val

    def _convert_to_magnitude_in(self, data_dict): 
        for label in data_dict:
            for i in range(len(data_dict[label])):
                arr = data_dict[label][i]
                arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])
        
    def _read_csv_files(self):
        data_dict = {c: [] for c in self.class_names}
        header_dict = {c: [] for c in self.class_names}

        columns = None
        for file in tqdm.tqdm(glob.glob(f"{self.path}/*.csv")):
            name = os.path.split(file)[-1][:-len(".csv")]
            if label := self.get_object_label(name, self.class_names, self.regexes):
                df = pd.read_csv(file)
                arr = df.to_numpy()
                header = arr[:,:3] #FIXME: check if only first 3 columns are header
                lc = arr[:,3:]

                data_dict[label].append(lc)
                header_dict[label].append(header)

                if columns is None:
                    columns = list(df.columns)
        for l in self.class_names:
            data_dict[l] = np.concatenate(data_dict[l])
            header_dict[l] = np.concatenate(header_dict[l])

        return data_dict, header_dict, columns

    def _push_to_max(self, example):
        
        y = example.copy()

        MAX_VALUE = 10_000_000
        y[y == 0] = MAX_VALUE

        minimal_y_value = np.amin(y)
        index_minimal = np.where(y == minimal_y_value)[0][0]
        
        y = np.roll(y, -index_minimal)
        
        y[y == MAX_VALUE] = 0
        
        return y

    def _fourier8(self,x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
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

    def _foufit(self, example):

        y = self._push_to_max(example)
        phases = np.linspace(0, 1, len(y), endpoint=False)

        non_zero = y != 0
        xs = phases[non_zero]
        ys = y[non_zero]

        params, params_covariance = optimize.curve_fit(self._fourier8, xs, ys, absolute_sigma=False, method="lm", maxfev=10000)
        std = np.sqrt(np.diag(params_covariance))

        return params, std

    def get_object_label(self, name, labels, regexes=None):
        def remove_extra_chars(string):
            return string.lower().replace("_", "").replace("-", "")
        
        if regexes:
            res = [l for l, r in zip(labels, regexes) if re.search(r, name, re.IGNORECASE)]
        else:
            res =  [l for l in labels if remove_extra_chars(l) in remove_extra_chars(name) and not "deb" in remove_extra_chars(name)]
        
        return None if res == [] else res[0]
    
    def get_pytorch_datasets(self):

        if  self.examples is None or self.class_names is None:
            raise Exception("No Data loaded yet. Please load data first.")
        
        (train_X, train_y), (val_X, val_y) = self.prepare_dataset(self.examples, self.labels)

        train_set = LCDataset(train_X, train_y)
        val_set = LCDataset(val_X, val_y)

        return train_set, val_set

