import numpy as np
from scipy import optimize

import tqdm

from src.nn.datasets.basic import BasicDataset


class FourierDataset(BasicDataset):

    std = 1
    mean = 0

    def __init__(self, data, labels, 
                 fourier, std, residuals, rms, amplitude, 
                 lc, reconstructed_lc, push_to_max, 
                 mode='val') -> None:
        
        self.use_fourier_params = fourier
        self.use_std = std
        self.use_residuals = residuals
        self.use_rms = rms
        self.use_amplitude = amplitude
        self.use_lc = lc
        self.use_reconstructed_lc = reconstructed_lc
        self.push_to_max = push_to_max
        
        self.data = []
        for example in tqdm.tqdm(data, desc="Computing Fourier"):
            self.data.append(self._preprocess_example(example))
        self.data = np.array(self.data).astype(np.float64)

        self.labels = labels
        self.offset = 0

        if len(data) < 1:
            return
        
        offset = 0
        if fourier:
            offset += 16
        if std:
            offset += 16
        if rms:
            offset += 1
        if amplitude:
            offset += 1

        self.offset = offset
        
        if offset > 0:
            if mode == 'train':
                self.compute_std_mean()
                            
            # self.data[:,:offset] = (self.data[:, :offset] - FourierDataset.mean) / FourierDataset.std

        # print(np.max(self.data, axis=0), np.min(self.data, axis=0), np.mean(self.data, axis=0))
    def normalize(self, example):
        if self.offset > 0:
            return (example[:self.offset] - FourierDataset.mean) / FourierDataset.std

    def compute_std_mean(self):
        FourierDataset.std = np.std(self.data[:,:self.offset], axis=0)
        FourierDataset.mean = np.mean(self.data[:, :self.offset], axis=0)

    def __getitem__(self, index):
        arr = self.data[index]
        label = self.labels[index]

        arr = self.normalize(arr)

        return arr, label

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
        
        y = example.copy()

        y[y == 0] = 10_000_000

        minimal_y_value = np.amin(y)
        index_minimal = np.where(y == minimal_y_value)[0][0]
        
        y = np.roll(y, -index_minimal)
        
        y[y == 10_000_000] = 0
        
        return y

    def _foufit(self, example):
        y = self._push_to_max(example)
        phases = np.linspace(0, 1, len(y), endpoint=False)

        non_zero = y != 0
        xs = phases[non_zero]
        ys = y[non_zero]

        params, params_covariance = optimize.curve_fit(self._fourier8, xs, ys, absolute_sigma=False, method="lm", maxfev=10000)
        std = np.sqrt(np.diag(params_covariance))

        y_hat = self._fourier8(phases, *params)
        
        amplitude = np.max(y_hat) - np.min(y_hat)
        
        residuals = np.abs(y - y_hat) / (amplitude + 1e-6)

        lc_normalized = y if self.push_to_max else example
        lc_normalized[lc_normalized != 0] = lc_normalized[lc_normalized != 0] - np.min(lc_normalized[lc_normalized != 0]) + 1e-5
        lc_normalized = lc_normalized / (amplitude + 1e-6)

        residuals[np.logical_not(non_zero)] = 0
        
        rms = np.sqrt(np.sum(residuals[non_zero]**2) / (residuals[non_zero].size-2))
        
        lc_reconstructed = (y_hat - np.min(y_hat) + 1e-5) / (amplitude + 1e-6)

        return params[1:], std[1:], residuals, lc_normalized, lc_reconstructed,  rms, amplitude


    def _preprocess_example(self, example):
        params, std, residuals, lc_normalized, lc_reconstructed, rms, amplitude = self._foufit(example)

        res = np.empty((0,))

        if  self.use_fourier_params:
            res = np.concatenate((res, params))
        
        if self.use_std:
            res = np.concatenate((res, std))

        if  self.use_rms:
            res = np.concatenate((res, [rms]))
        
        if  self.use_amplitude:
            res = np.concatenate((res, [amplitude]))

        if self.use_residuals:
            res = np.concatenate((res, residuals))

        if  self.use_lc:
            res = np.concatenate((res, lc_normalized))
            
        if  self.use_reconstructed_lc:
            res = np.concatenate((res, lc_reconstructed))

        return res

