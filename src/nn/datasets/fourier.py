import numpy as np
from scipy import optimize

from src.nn.datasets.basic import BasicDataset


class FourierDataset(BasicDataset):

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

