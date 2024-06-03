import os
import numpy as np

def load_multi_array(path):
    with open(path, 'rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))

    return out