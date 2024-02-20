from collections import defaultdict
from typing import Callable, Dict, List
import pandas as pd
import numpy as np
import tqdm
import glob
import os
import re

from collections import defaultdict

def load_data(path, labels, regexes=None, convert_to_mag=False):

    df = pd.read_csv(path, index_col=0)
    data = defaultdict(list)

    for name in df["Object name"].unique():

        label = None

        for i in range(len(labels)):
            if re.search(regexes[i], name):
                label = labels[i]
                break
        if label is None:
            continue

        df_object = df[df["Object name"] == name]
        object_IDs = df_object["Object ID"].unique()

        for object_ID in object_IDs:
            df_object_ID = df_object[df_object["Object ID"] == object_ID]

            arr = df_object_ID.to_numpy()[:, 4:]
            if convert_to_mag:
                arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])
            data[label].append(arr) 
    
    return data 

def get_object_label(name, labels, regexes=None):
    name2 = name.lower().replace("_", "").replace("-", "")
    for i, label in enumerate(labels):
        label2 = label.lower().replace("_", "").replace("-", "")

        if regexes is not None:
            if re.search(regexes[i], name, re.IGNORECASE):
                return label            
        else:
            if label2 in name2.lower() and not "deb" in name2.lower():
                return label
    return None

def get_non_zero_ratio(data: np.array) -> np.array:
    return np.sum(data != 0, axis=1)

def dict_map(data: Dict, func: Callable) -> Dict:
    return { key: func(value) for key, value in data.items()}

def sort_func(arr):
    stats = np.sum(arr != 0, axis=1)
    indices = np.argsort(-stats)
    return arr[indices]

def get_representants(arr):

    best = arr[:3]
    worst = arr[-3:]

    mid = len(arr)//2
    middle = arr[mid-1:mid+2]

    data = np.concatenate((best, middle, worst))
    titles = [f'Signal ratio: {x/300 * 100:.2f}%' for x in np.sum(data != 0, axis=1)]

    return data, titles

def get_stats(arr):
    stats = np.sum(arr != 0, axis=1) / 300 * 100
    return stats