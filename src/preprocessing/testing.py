#%%
import numpy as np

#%%
def SMA(light_curve, size, window_size):
    N = len(light_curve)
    step = 1 / (size + window_size)

    data = np.zeros(size)
    window_t = step * window_size

    t = window_t
    idx1, idx2 = 0, 0

    for i in range(size):

        if idx1 == len(light_curve):
            break

        while idx2 < N and light_curve[idx2][1] <= t:
            idx2 += 1
        while idx1 < N and light_curve[idx1][1] < t - window_t:
            idx1 += 1

        window_data = light_curve[idx1:idx2, 0]
        if len(window_data):
            data[i] = np.sum(window_data) / window_size
        t += step

    return data
#%%

def SMA2(light_curve, size, window_size):
    N = len(light_curve)
    step = 1 / (size + window_size)

    new_data = np.zeros(size+window_size)
    for i, (value, time) in enumerate(light_curve):
        idx = min(int(np.round(time / step)), size + window_size -1)
        new_data[idx] += value / window_size

    data = np.array([np.sum(new_data[i:i+window_size]) for i in range(size)])

    return data
#%%
import time

np.random.seed(42)
lcs = np.random.rand(100,2000,2)

s = time.time()
for lc in lcs:
    r = SMA(lc, 200, 5)
e = time.time() - s
print("SMA time: ", e/100)


s = time.time()
for lc in lcs:
    r = SMA2(lc, 200, 5)
e = time.time() - s
print("SMA time: ", e/100)
#
# lc_simple = np.array([[1,0],[1,1/7],[1,2/7],[1,3/7],[1,4/7],[1,5/7],[1,6/7]])
# lc = np.random.rand(10,2)
# lc = lc[np.argsort(lc[:,1])]
# print(lc[:, 1]*7)
#
# r2 = SMA2(lc, 7, 3)
# print(r2)

# r1 = SMA(lc, 7, 3)
# print(r1)