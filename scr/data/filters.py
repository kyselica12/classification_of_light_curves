import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from functools import partial

from data.data_load import dict_map
from config import FilterConfig

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

def filter_data(data, filter_config:FilterConfig):

    f_continous = partial(get_filter_continuous, n_bins=filter_config.n_bins, 
                                                gap=filter_config.n_gaps, 
                                                continous_gap=filter_config.gap_size)
    f_ratio = partial(get_filter_ratio, ratio=filter_config.non_zero_ratio)
    app_filters_p = partial(apply_filters, filters_f=[f_continous, f_ratio], operation="AND")

    filtered_data = dict_map(data, app_filters_p)

    return filtered_data