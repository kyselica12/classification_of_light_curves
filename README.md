# Sattelite's shape prediction

## Data

Input data are light curves from **MMT** database. Data are collected using sctip. Downloaded data folder contains two main folder:
1) **period** - contains information about created observation of an object and its period
2) **objects** - contains observations for specific object


Data are processed by ```preprocessing/main.py``` script

## Preprocessing

Process raw data from MMT database to *numpy_arrays**. Subdivides light curve by period and reshape newly created light curves to 
the same size.

Resulting arrays are stored in **"<object name> multi_array.npy"**. **multi_array.npy** file can be loaded 
with `load_multi_array.load_multi_array` method. Resulting array's shape is **n x l**, where **n** is the 
number of light curves and **l** is length of those curves.

###Arguments
- `-i` / `--input` - path to input folder containing period and objects directories
- `-o` / `--output` - path to output directory where resulting multi_arrays are stored
- `-l` / `--size` - size of single light curve
- `-t` / `--len-threshold` - number **0 <= t <= 1**. Light curves with less than **t x l** nonzero values will be discarded.
- `-w` / `--window-size` - size of the window for simple moving average method

**Run:**

```python3 main.py -i "input_folder" -o "output_folder" -l 300 -t 0 -w 1```



# TODOs

- Try model with mormalized data
- Different filter values
- Data augmentation

- Data modeling