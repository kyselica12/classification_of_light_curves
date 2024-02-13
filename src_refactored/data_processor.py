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
        self.labels = data_config.labels
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

    def _generate_hash(self):
        hash_text = f'{self.labels}_{self.regexes}_{self.convert_to_mag}_{self.filter_config}'
        return hashlib.md5(hash_text.encode()).hexdigest()

    def save_data_MMT(self):
        hash = self._generate_hash()
        os.makedirs(f'{self.output_path}/{hash}', exist_ok=True)

        np.savetxt("examples.txt", self.examples)
        np.savetxt("labels.txt", self.labels)

    def load_data_MMT(self):
        hash = self._generate_hash()
        path = f'{self.output_path}/{hash}'

        examples = np.loadtxt(f'{path}/examples.txt')
        labels = np.loadtxt(f'{path}/labels.txt')

        self.examples = examples
        self.labels = labels

    def load_raw_data_MMT(self):
        data_dict, header_dict, columns = self._read_csv_files()

        if self.convert_to_mag:
            self._convert_to_magnitude_in(data_dict)

        if self.filter_config:
            data_dict = filter_data(data_dict, self.filter_config)
            
        columns += [f"Fourier coef " for i in range(16)]
        print("Computing Fourier....")

        fourier_coefs = []
        for label, data in data_dict.items():
            for d in tqdm.tqdm(data, desc=f"Fourier for class {label}"):
                fourier_coefs.append(self._foufit(d))

        self.labels = np.array([idx for idx,l in enumerate(data_dict) for i in range(len(data_dict[l]))])
        examples = np.array([d for ds in data_dict.values() for d in ds])
        self.examples = np.concatenate((examples, np.array(fourier_coefs)), axis=1)

    def prepare_dataset(self, examples, labels, seed=None):

        N = len(examples)
        data = []
        lc = examples[:,:LC_SIZE]
        fc_rms = examples[:,LC_SIZE:]
        fc = fc_rms[:,:2*FOURIER_N+1]
        rms = fc_rms[:,2*FOURIER_N+1:]

        phases = np.linspace(0, 1, LC_SIZE, endpoint=False)
        print(fc.shape, rms.shape, lc.shape)
        y_hat = np.array([self._fourier8(phases, *(list(fc[i]))) for i in range(N)])
        
        amplitude = np.max(y_hat, axis=1) - np.min(y_hat, axis=1)
        residuals = np.abs(lc - y_hat) / (amplitude + 1e-6)
        non_zero = lc != 0
        
        if self.use_lc:
            data.append(lc[non_zero] - np.min(lc[non_zero], axis=1) + 1e-6 / (amplitude + 1e-6))
        if self.use_reconstructed:
            recontructed_lc = (y_hat - np.min(y_hat,axis=1) + 1e-6) / (amplitude + 1e-6)
            data.append(recontructed_lc)
        if self.use_residuals:
            data.append(residuals)
        if self.use_fourier:
            data.append(fc[:,1:])
        if self.use_std:
            data.append(rms[:,1:])
        if self.use_rms:
            rms = np.sqrt(np.sum(residuals[non_zero]**2) / (residuals[non_zero].size-2))
            data.append(rms)
        if self.use_amplitude:
            data.append(amplitude / self.max_amplitude)

        X = np.concatenate(tuple(data), axis=1)
        y = np.array(labels)

        (train_X, train_y),(val_X, val_y) = self.split_dataset(X,y,self.validation_split, seed)

        return (train_X, train_y), (val_X, val_y)

    def split_dataset(self, X,y, split=0.1, seed=None):
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

    def _convert_to_magnitude_in(self, data_dict): 
        for label in data_dict:
            for i in range(len(data_dict[label])):
                arr = data_dict[label][i]
                arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])
        
    def _read_csv_files(self):
        data_dict = defaultdict(list)
        header_dict = defaultdict(list)
        columns = None
        for file in tqdm.tqdm(glob.glob(f"{self.path}/*.csv")):
            name = os.path.split(file)[-1]
            if label := self.get_object_label(name, self.labels, self.regexes):
                df = pd.read_csv(file)
                arr = df.to_numpy()
                header = arr[:,:4]
                lc = arr[:,:4]

                data_dict[label].append(lc)
                header_dict[label].append(header)

                if columns is None:
                    columns = list(df.columns)
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

        if  self.examples is None or self.labels is None:
            raise Exception("No Data loaded yet. Please load data first.")
        
        (train_X, train_y), (val_X, val_y) = self.prepare_dataset(self.examples, self.labels)

        train_set = LCDataset(train_X, train_y)
        val_set = LCDataset(val_X, val_y)

        return train_set, val_set

