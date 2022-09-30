import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from functools import partial

from data.data_load import dict_map
from config import N_BINS, N_GAPS, GAP_SIZE, NON_ZERP_RATIO, RMS_RATIO

def get_filter_continuous(data, n_bins=10, gap=0, continous_gap=3):
    N = 300 // n_bins

    x = np.resize(data, (data.shape[0], n_bins, N))
    bins = np.sum(x, axis=2) != 0
    bins_sum = np.sum(bins, axis=1)

    res = bins_sum >= (n_bins - gap)

    if continous_gap > 0:
        continous_gaps = sliding_window_view(bins, window_shape=continous_gap+1, axis=1)
        continous_gaps_ok = np.all(np.sum(continous_gaps, axis=2) != 0, axis=1)

        res = np.logical_and(res, continous_gaps_ok)

    return res

def get_filter_ratio(data, ratio=0.5):

    x = np.sum(data != 0, axis= 1) / 300
    return x >= ratio

def apply_filters(data, filters_f, operation="AND"):

    f_res = None

    for f in filters_f:
        if f_res is None:
            f_res = f(data)
        else:
            if operation == "AND":
                f_res = np.logical_and(f(data), f_res)
            else:
                print(":)")
                f_res = np.logical_or(f(data), f_res)
    
    return data[f_res]

def apply_sequential_filters(data, filters):

    for f in filters:
        ok = f(data)
        data = data[ok]

    return data


def filter_data(data):

    filters = []
    filters.append(partial(get_filter_continuous, n_bins=N_BINS, 
                                                gap=N_GAPS, 
                                                continous_gap=GAP_SIZE))
    filters.append(partial(get_filter_ratio, ratio=NON_ZERP_RATIO))

    if RMS_RATIO != 0:
        filters.append(partial(get_rms_filter, rms_ratio=RMS_RATIO))
    # app_filters_p = partial(apply_filters, filters_f=filters, operation="AND")
    app_filters_p = partial(apply_sequential_filters, filters=filters)
    filtered_data = dict_map(data, app_filters_p)

    return filtered_data


def get_rms_filter(data_list, rms_ratio=0.5):
    
    ok = np.zeros((data_list.shape[0])).astype(bool)
    print(data_list.shape)
    
    for i, data in enumerate(data_list):
        indices = data != 0
        x = np.linspace(0,len(data), endpoint=False, num=len(data))[indices]
        y = data[indices]
        ampl = np.max(y) - np.min(y)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            p30 = np.poly1d(np.polyfit(x, y, 30))
            yy = p30(x)
        
        del p30
        rms = np.sqrt(np.mean((y-yy)**2))

        ok[i] = rms / (ampl+10**(-6)) <= rms_ratio

    return ok