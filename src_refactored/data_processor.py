import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import optimize
import tqdm

from src_refactored.configs import DataConfig
from src_refactored.filters import filter_data


class DataProcessor:
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.labels = self.data_config.labels
        self.regexes = self.data_config.regexes
        self.path = self.data_config.path
        self.output_path = self.data_config.output_path
        self.validation_split = self.data_config.validation_split
        self.number_of_training_examples_per_class =data_config.number_of_training_examples_per_class
        self.convert_to_mag = data_config.convert_to_mag
        self.filter_config = data_config.filter_config

        self.use_fourier = data_config.fourier
        self.use_std = data_config.std
        self.use_residuals = data_config.residuals
        self.use_rms = data_config.rms
        self.use_amplitude = data_config.amplitude
        self.use_lc = data_config.lc
        self.use_reconstructed = data_config.reconstructed_lc
        self.push_to_max = data_config.push_to_max

        self.lc_std = 1
        self.lc_mean = 0
        self.f_std = 1
        self.f_mena = 0

    def create_dataset(self):
        data_dict, header_dict, size = self._load_csv()
        
        if self.filter_config:
            data_dict = filter_data(data_dict, self.filter_config)

        data = []
        for d in data_dict.values():
            data += d
        names = ["label"] + ["Object ID", "Track ID", "Phase", "Start", "Period"]
        names += [str(i) for i in range(size)]
        names += [f"fs_{i}" for i in range(16)]
        names += [f"std_{i}" for i in range(16)]

        for k, v in data_dict.items():
            header = header_dict[k]

        for example in tqdm.tqdm(data, desc="Computing Fourier"):
            coefs, std, residuals, lc_norm, lc_rec, rms, amp  = self._foufit(example)
         

        fourier_coeficients = np.array(fourier_coeficients).astype(np.float64)
        


    def _load_csv(self):
        df = pd.read_csv(self.path)
        data_dict = defaultdict(list)
        header_dict = defaultdict(list)
        size = 0

        for name in df["Object name"].unique():
            
            if label := self.get_object_label(name, self.labels, self.regexes):
                object_data = df[df["object name"] == name]  

                for object_ID in object_data["Object ID"].unique():
                    arr = object_data[object_data["Object ID"] == object_ID].to_numpy()
                    data_arr = arr[:, 4:]
                    header_arr = arr[:,:4]

                    size = len(data_arr)
                    # arr = object_data[object_data["Object ID"] == object_ID].to_numpy()[:, 4:] # remove first 4 columns

                    if self.convert_to_mag:
                        data_arr[data_arr != 0] = -2.5 * np.log10(data_arr[data_arr != 0])

                    data_dict[label].append(data_arr)
                    header_dict[label].append(header_arr)
            
        return data_dict, header_dict, size

    def _push_to_max(self, example):
        
        y = example.copy()

        y[y == 0] = 10_000_000

        minimal_y_value = np.amin(y)
        index_minimal = np.where(y == minimal_y_value)[0][0]
        
        y = np.roll(y, -index_minimal)
        
        y[y == 10_000_000] = 0
        
        return y

    def _foufit(self, example):
        def fourier8(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
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

        y = self._push_to_max(example)
        phases = np.linspace(0, 1, len(y), endpoint=False)

        non_zero = y != 0
        xs = phases[non_zero]
        ys = y[non_zero]

        params, params_covariance = optimize.curve_fit(fourier8, xs, ys, absolute_sigma=False, method="lm", maxfev=10000)
        std = np.sqrt(np.diag(params_covariance))

        # y_hat = fourier8(phases, *params)
        # 
        # amplitude = np.max(y_hat) - np.min(y_hat)
        # 
        # residuals = np.abs(y - y_hat) / (amplitude + 1e-6)
        #
        # lc_normalized = y if self.push_to_max else example
        # #TODO better normalization process???
        # lc_normalized[lc_normalized != 0] = lc_normalized[lc_normalized != 0] - np.min(lc_normalized[lc_normalized != 0]) + 1e-5
        # lc_normalized = lc_normalized / (amplitude + 1e-6)
        #
        # residuals[np.logical_not(non_zero)] = 0
        # 
        # rms = np.sqrt(np.sum(residuals[non_zero]**2) / (residuals[non_zero].size-2))
        # 
        # lc_reconstructed = (y_hat - np.min(y_hat) + 1e-5) / (amplitude + 1e-6)

        return params[1:], std[1:]

    def get_object_label(self, name, labels, regexes=None):
        def remove_extra_chars(string):
            return string.lower().replace("_", "").replace("-", "")

        if regexes:
            return [l for l, r in zip(labels, regexes) if re.search(r, name, re.IGNORECASE)][0]

        return [l for l in labels if remove_extra_chars(l) in remove_extra_chars(name) and not "deb" in remove_extra_chars(name)][0]



