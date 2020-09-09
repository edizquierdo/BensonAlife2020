##################################################
# figure 3
# connected swarm of individual fitness of multifunc ensemble
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def connected_swarms(dir):
    plt.figure(figsize=[5, 3])

    # load data
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()
    print("Found {} files in {}".format(len(files), dir))
    dat = []
    for i, file in enumerate(files):
        fits = np.load(file)
        if np.prod(fits) > 0.8:
            fits = np.round(fits, decimals=4)
            print(i, file, fits, np.prod(fits))
            dat.append(np.concatenate([[i], ["IP"], [fits[0]]]))
            dat.append(np.concatenate([[i], ["CP"], [fits[1]]]))
            dat.append(np.concatenate([[i], ["LW"], [fits[2]]]))

    # make DataFrame and plot
    df = pd.DataFrame(dat, columns=["id","task", "fitness"])
    ax = sns.swarmplot(
        x="task",
        y="fitness",
        hue="id",
        data=df,
    )
#    ax._legend.remove()
    ax.set_yticks(ax.get_yticks()[::5])

    # plot connecting lines
    x1, y1 = np.array(ax.collections[0].get_offsets()).T
    x2, y2 = np.array(ax.collections[1].get_offsets()).T
    x3, y3 = np.array(ax.collections[2].get_offsets()).T
    ind = 0
    for xi, xj, xk, yi, yj, yk in zip(x1, x2, x3, y1, y2, y3):
        if yi == np.max(y1):  # best of the best
            print("Max == ", yi, yi * yj * yk)
            plt.plot([xi, xj], [yi, yj], "black")
            plt.plot([xj, xk], [yj, yk], "black")
        else:
            plt.plot([xi, xj], [yi, yj], "gray", alpha=0.3)
            plt.plot([xj, xk], [yj, yk], "gray", alpha=0.3)
        ind += 1

    plt.tight_layout()
    plt.savefig("figure_3.pdf")
    plt.show()


connected_swarms("../New")
