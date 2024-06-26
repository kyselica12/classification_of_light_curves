from collections import defaultdict
from functools import partial
import os
import random
import hashlib
import re
import glob
import numpy as np
import pandas as pd
import pywt
from scipy import optimize
import tqdm
from strenum import StrEnum

from src.configs import DataConfig, LC_SIZE, FOURIER_N, SplitStrategy
from src.configs import DataType as DT, DatasetType as DST
from src.data.filters import filter_data
from src.data.dataset import LCDataset

LABELS = "labels"
HEADERS = "headers"


class DataProcessor:
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.class_names = data_config.class_names
        self.n_classes = len(self.class_names)
        self.regexes = data_config.regexes

        self.path = data_config.path
        self.test_path = data_config.validation_path
        self.output_path = data_config.output_path
        self.artificial_data_path = data_config.artificial_data_path

        self.validation_split = data_config.validation_split
        self.number_of_training_examples_per_class= data_config.number_of_training_examples_per_class
        self.convert_to_mag = data_config.convert_to_mag
        self.max_amplitude = data_config.max_amplitude
        self.filter_config = data_config.filter_config
        self.split_strategy = data_config.split_strategy
        self.seed = data_config.seed

        self.wavelet_start = data_config.wavelet_start_scale
        self.wavelet_end = data_config.wavelet_end_scale
        self.wavelet_step = data_config.wavelet_scales_step
        self.max_wavelet_scales = 10
        self.wavelet_name = data_config.wavelet_name
        self.lc_shifts = data_config.lc_shifts

        self.use_data_types = data_config.data_types
        self.data = {}
        self.test_data = {}
        self.artificial_data = {}

        hash_text = f'{self.class_names}_{self.regexes}_{self.convert_to_mag}_{self.filter_config}'
        hash_text += f"_{self.wavelet_name}"
        # hash_text += self.path + str(self.output_path) + self.test_path
        self.hash = hashlib.md5(hash_text.encode()).hexdigest()
        print(f"Hash: {self.hash}")
    
    def data_shape(self):
        shape = 0
        for t in self.use_data_types: 
            match t:
                case DT.FS | DT.STD:
                    shape += FOURIER_N*2 -1 
                case DT.WAVELET:
                    shape += (self.wavelet_end - self.wavelet_start + 1) // self.wavelet_step * LC_SIZE
                case DT.AMPLITUDE | DT.RMS: 
                    shape += 1
                case DT.LC:
                    shape += LC_SIZE * (1 + self.lc_shifts)
                case DT.RECONSTRUCTED_LC:
                    shape += LC_SIZE
                case DT.RESIDUALS:
                    shape += LC_SIZE
                case _:
                    raise ValueError(f"Data type {t} not recognized")

        return shape
        

    def save_data(self, type=DST.TRAIN):
        path = f"{self.output_path}/{self.hash}{'' if type==DST.TRAIN else '/'+type}"

        data = self.get_data_by_type(type)

        os.makedirs(path, exist_ok=True)

        for t in data:
            if data[t] is not None:
                self._save_data_type(t, type)
            else:
                print(f"Data type {t} is None. Skipping...")

    def get_data_by_type(self, type):
        match type:
            case DST.TRAIN:
                data = self.data
            case DST.TEST:
                data = self.test_data
            case DST.ARTIFICIAL:
                data = self.artificial_data
        return data

    def _save_data_type(self, t, type=DST.TRAIN):
        directory = f"{self.output_path}/{self.hash}{''if type==DST.TRAIN else '/'+type}"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        filename = f"{directory}/{t}"

        data = self.get_data_by_type(type)
        
        if t == DT.WAVELET:
            filename += f"_{self.wavelet_name}_{self.max_wavelet_scales}"
        np.save(filename+".npy", data[t])
    
    def load_data_from_file(self, type=DST.TRAIN):
        to_load = list([LABELS, HEADERS, DT.LC, DT.AMPLITUDE,
                       *self.use_data_types])

        data = self.get_data_by_type(type)

        for t in to_load:
            if t in data: continue
            if (arr := self._load_data_type(t, type)) is not None:
                data[t] = arr
                if t == DT.WAVELET:
                    data[t] = data[t][:,self.wavelet_start-1:self.wavelet_end:self.wavelet_step, :]

    def _load_data_type(self, t, type=DST.TRAIN):
        # filename = f"{self.output_path}/{self.hash}/{t}"
        filename = f"{self.output_path}/{self.hash}{''if type==DST.TRAIN else '/'+type}/{t}"
        if t == DT.WAVELET:
            filename += f"_{self.wavelet_name}_{self.max_wavelet_scales}"

        filename += ".npy"
        
        if os.path.exists(filename):
            data =  np.load(filename)
            return data

        print(f"File {filename} not found. Skipping...")
        return None

    def unload(self):
        self.data.clear()
        self.test_data.clear()
        self.artificial_data.clear()

    
    def create_dataset_from_csv(self, type=DST.TRAIN):
        def read_dataset(data, read_function):
            print(f"Reading csv files....")
            data_dict, header_dict, _ = read_function()

            if self.convert_to_mag:
                self._convert_to_magnitude_in(data_dict)

            if self.filter_config:
                data_dict, header_dict = filter_data(data_dict, header_dict, self.filter_config)
                for l in data_dict:
                    print(f"After filtration {len(data_dict[l])} examples for class {l}")

            data[LABELS] = np.array([i for i, l in enumerate(self.class_names) for _ in range(len(data_dict[l]))])
            data[HEADERS] = np.concatenate([header_dict[l] for l in self.class_names])
            lc = data[DT.LC] = np.concatenate([data_dict[l] for l in self.class_names])

            print("Computing Fourier Series....")
            fc_std = np.array([self._foufit(d) for d in tqdm.tqdm(data[DT.LC])])
            fourier_coefs = data[DT.FS] = fc_std[:,0]
            data[DT.STD] = fc_std[:,1]

            phases = np.linspace(0, 1, LC_SIZE, endpoint=False)
            y_hat = np.array([self._fourier8(phases, *(list(c))) for c in fourier_coefs])
            
            amplitude = data[DT.AMPLITUDE] = (np.max(y_hat, axis=1) - np.min(y_hat, axis=1)).reshape(-1,1)
            residuals = data[DT.RESIDUALS] =  np.abs(lc - y_hat) / (amplitude + 1e-6)

            data[DT.RECONSTRUCTED_LC] = (y_hat - np.min(y_hat,axis=1, keepdims=True) + 1e-6) / (amplitude + 1e-6)
            data[DT.RMS] = np.sqrt(np.sum(residuals**2,axis=1) / (np.sum(residuals != 0, axis=1)-2 + 1e-6)).reshape(-1,1)

            # self._compute_wavelet_transform(self.wavelet_scales, self.wavelet_step)
            data[DT.WAVELET] = self._compute_wavelet_transform(data, 1, self.max_wavelet_scales, 1)


        print("Reading training data....")
        match type:
            case DST.TRAIN:
                read_dataset(self.data, partial(self._read_csv_files, self.path))
                # TODO align phases for MIXUP
                # self._align_phases(self.data[DT.LC])
            case DST.TEST:
                read_dataset(self.test_data, self._read_SDLCD_csv)
                # TODO align phases for MIXUP
                # self._align_phases(self.test_data[DT.LC])
            case DST.ARTIFICIAL:
                read_dataset(self.artificial_data, partial(self._read_csv_files, self.artificial_data_path))
            case _:
                raise ValueError(f"Dataset type {type} not recognized. Use one of: 'train', 'test', 'artificial'")
        # read_dataset(self.data, partial(self._read_csv_files, self.path))
        # self._align_phases(self.data[DT.LC])

        # if self.artificial_data_path is not None:
        #     print("Reading artificial data....")
        #     read_dataset(self.artificial_data, partial(self._read_csv_files, self.artificial_data_path))
        #     self._align_phases(self.artificial_data[DT.LC], self.data[DT.LC][0])

        # if self.test_path != "":
        #     print("Reading test data....")
        #     read_dataset(self.test_data, self._read_SDLCD_csv)

    
    def _align_phases(self, lcs, first=None):
        total_err = np.sum([np.linalg.norm(lcs[i] - lcs[i-1]) for i in range(1, len(lcs))])
        print(f"Total error before aligning: {total_err}")
        u = lcs[0] if first is None else first
        start = 1 if first is None else 0
        for i in tqdm.tqdm(range(start, len(lcs)), desc="Aligning phases"):
            # u = lcs[i-1]
            v = lcs[i]
            lcs[i] = min([np.roll(v, r, axis=0) for r in range(0, v.shape[-1])],
                         key=lambda vv: np.linalg.norm(vv-u))
            u = v

        total_err = np.sum([np.linalg.norm(lcs[i] - lcs[i-1]) for i in range(1, len(lcs))])
        print(f"Total error after aligning: {total_err}")

    def to_output_format(self,data):
        examples = []
        for t in self.use_data_types:
            match t:
                case DT.LC:
                    lc = data[t].copy()
                    lc[lc == 0] = np.nan
                    lc = (lc - np.nanmin(lc, axis=1, keepdims=True) + 1e-6) / (data[DT.AMPLITUDE].reshape(-1,1) + 1e-6)
                    lc[np.isnan(lc)] = 0
                    examples.append(self._compute_lc_shifts(lc))
                case DT.FS | DT.STD:
                    examples.append(data[t][:,1:]) 
                case DT.AMPLITUDE:
                    examples.append((data[t] / self.max_amplitude).reshape(-1,1))
                case DT.WAVELET:
                    if data[t].shape[1] > (self.wavelet_end - self.wavelet_start + 1) / self.wavelet_step:
                        data[t] = data[t][:,self.wavelet_start-1:self.wavelet_end:self.wavelet_step, :]
                    examples.append(data[t].reshape(len(data[t]),-1))
                case _:
                    examples.append(data[t])

        return examples
    def prepare_dataset(self):

        examples = self.to_output_format(self.data)
        X = np.concatenate(tuple(examples), axis=1)
        y = self.data[LABELS]

        train_set, val_set = self.split_dataset(X,y)
        test_set = (None, None)

        if self.artificial_data != {}:
            (train_X, train_y) = train_set
            val_set = train_set # FOR now to train only on artificial and validate on real
            examples = self.to_output_format(self.artificial_data)
            synthetic_X = np.concatenate(tuple(examples), axis=1)
            train_X = np.concatenate((train_X, synthetic_X), axis=0)
            train_y = np.concatenate((train_y, self.artificial_data[LABELS]))
            # train_X = synthetic_X
            # train_y = self.artificial_data[LABELS]
            train_set = (train_X, train_y)
        
        
        if self.test_data != {}:
            test_examples = self.to_output_format(self.test_data)
            X_test = np.concatenate(tuple(test_examples), axis=1)
            y_test = self.test_data[LABELS]
            from collections import Counter
            c = Counter(y_test)
            print(c)
            test_set = (X_test, y_test)
        
        return (train_set, val_set, test_set)

    
    def _compute_lc_shifts(self, lc):
        if self.lc_shifts == 0:
            return lc

        shifts = [lc.copy()]
        shift_size = LC_SIZE//(self.lc_shifts+1)
        for i in range(self.lc_shifts):
             shifts.append(np.roll(lc, i * shift_size , axis=1))
        return np.concatenate(shifts, axis=1)
        

    def _compute_wavelet_transform(self,data, start, end, step):
        print("Computing Continuous Wavelet Transform....")
        lc = data[DT.LC].copy()
        lc[lc == 0] = data[DT.RECONSTRUCTED_LC][lc == 0]
        lc = (lc - np.nanmin(lc, axis=1, keepdims=True) + 1e-6) / (data[DT.AMPLITUDE].reshape(-1,1) + 1e-6)

        scales = np.arange(start, end+1, step)
        coef, _ = pywt.cwt(lc ,scales,self.wavelet_name)
        coef = coef.transpose(1,0,2) # (N, scales, LC_SIZE)
        return coef
                    
    def split_dataset(self, X, y):
        match self.split_strategy:
            case SplitStrategy.RANDOM:
                split = self._split_random(X,y,self.validation_split, self.seed)
            case SplitStrategy.OBJECT_ID | SplitStrategy.TRACK_ID:
                header_idx = 0 if self.split_strategy == "objectID" else 1
                split = self._split_by_object_or_track(X, y, self.data[HEADERS], 
                                                   self.number_of_training_examples_per_class,
                                                   self.validation_split,
                                                   split_on_header_idx=header_idx)
            case SplitStrategy.NO_SPLIT:
                split = ((X,y),([], []))
            case _:
                raise ValueError(f"Split strategy {self.split_strategy} not recognized. Use one of: 'random', 'objectID', 'trackID'")

        (train_X, train_y),(val_X, val_y) = split

        return (train_X, train_y),(val_X, val_y)  


    def _split_random(self, X,y, split=0.1, seed=None):
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

    def _split_by_object_or_track(self, X, Y, headers, k, split=0.1, split_on_header_idx=0):

        train_X = np.empty((0, *X.shape[1:]))
        train_y = np.empty((0,))
        val_X = np.empty((0, *X.shape[1:]))
        val_y = np.empty((0,))

        for label in range(len(self.class_names)):
            mask = Y == label
            x = X[mask]
            h = headers[mask][:, split_on_header_idx]

            x_obj = [x[h==idx] for idx in np.unique(h)]
            sizes = list(map(len, x_obj))
            N = sum(sizes)

            indices = np.argsort(-np.array(sizes))
            total = 0

            for idx in tqdm.tqdm(indices, desc=f"Splitting {self.class_names[label]}"):
                if (sizes[idx] + total < k*1.1 and sizes[idx] + total < N * (1-split)) or \
                    (total == 0 and sizes[idx] + total < N * (1-split)):
                    total += sizes[idx]
                    train_X = np.concatenate((train_X, x_obj[idx]))
                    train_y = np.concatenate((train_y, np.ones((x_obj[idx].shape[0],))*label))
                else:
                    val_X = np.concatenate((val_X, x_obj[idx]))
                    val_y = np.concatenate((val_y, np.ones((x_obj[idx].shape[0],))*label))

        return (train_X, train_y), (val_X, val_y)
    
    def _convert_to_magnitude_in(self, data_dict): 
        for label in data_dict:
            for i in range(len(data_dict[label])):
                arr = data_dict[label][i]
                arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])

    def _read_SDLCD_csv(self):
        df = pd.read_csv(self.test_path)
        arr = df.to_numpy()
        columns = list(df.columns)

        data_dict = {c: [] for c in self.class_names}
        header_dict = {c: [] for c in self.class_names}

        for x in arr:
            name = x[0].replace(' ', '_')
            if label := self.get_object_label(name, self.class_names, self.regexes):
                data_dict[label].append(x[4:])
                header = x[1:4]
                # for i in range(26):
                    # header[0] = header[0].replace(chr(ord('A')+i), str(i))
                header_dict[label].append(header)
            else:
                print(f"Object {name} not recognized. Skipping...")
        
        for l in self.class_names:
            data_dict[l] = np.array(data_dict[l]).reshape(-1, LC_SIZE).astype(np.float32)
            header_dict[l] = np.array(header_dict[l]).reshape(-1, 3).astype(np.float32)

        return data_dict, header_dict, columns
        
       
    def _read_csv_files(self, path):
        print(self.class_names)
        data_dict = {c: [] for c in self.class_names}
        header_dict = {c: [] for c in self.class_names}

        columns = None
        for file in tqdm.tqdm(glob.glob(f"{path}/*.csv")):
            name = os.path.split(file)[-1][:-len(".csv")]
            if label := self.get_object_label(name, self.class_names, self.regexes):
                df = pd.read_csv(file)
                arr = df.to_numpy()
                header = arr[:,:3]
                lc = arr[:,3:]

                data_dict[label].append(lc)
                header_dict[label].append(header)

                if columns is None:
                    columns = list(df.columns)

        for l in self.class_names:
            print(len(data_dict[l]), l)
            data_dict[l] =  np.concatenate(data_dict[l])
            header_dict[l] =  np.concatenate(header_dict[l])
            print(f"Loaded {len(data_dict[l])} examples for class {l}")

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

        if  len(self.data) < 3:
            raise Exception("No Data loaded yet. Please load data first.")
        
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = self.prepare_dataset()

        train_set = LCDataset(train_X, train_y, self.n_classes, self.data_config.train_augmentations )
        val_set = LCDataset(val_X, val_y, self.n_classes)
        print(train_set.use_mixup, train_set.use_cyclic_augmentation)
        test_set = None
        if self.test_path != "":
            test_set = LCDataset(test_X, test_y, self.n_classes)
            from collections import Counter
            print( Counter(test_set.labels))

        return train_set, val_set, test_set

