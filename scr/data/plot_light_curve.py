import warnings
import matplotlib.pyplot as plt
from math import ceil
import numpy as np


def plot_curves(data, n_cols=2, save_path=None, titles=None, fit=False):

    n_rows =  ceil(len(data)/n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    fig.set_size_inches(2.5*n_cols, 2.7*n_rows)

    for i in range(n_rows):
        for j in range(n_cols):
            lc = data[i*n_cols+j]
            x = np.linspace(0,1, endpoint=True, num=300)[lc != 0]
            y = lc[lc != 0]
            axs[i,j].scatter(x, y, s=1)

            if fit:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', np.RankWarning)
                    p30 = np.poly1d(np.polyfit(x, y, 30))
                    axs[i, j].plot(x, p30(x), '-')

            if titles:
                axs[i,j].title.set_text(titles[i*n_cols+j])

    if save_path:
        fig.savefig(save_path, dpi=500)

    plt.tight_layout()
    plt.show()
    