##################################################
# figure 1
# gens vs fitnness for 3 tasks individually
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def fitness_traces(dir, color, label):
    files = glob.glob(os.path.join(dir, "best_history*.npy"))
    print("Found {} files in {}".format(len(files), dir))
    bfs = np.zeros(len(files))
    for i, file in enumerate(files):
        bf = np.load(file)
        bfs[i] = bf[-1]
        plt.plot(bf, color=color, label=label)
        label = None

    print("Number of runs > 0.93 = ", len(bfs[bfs >= 0.93]))
    print("Best of all runs = ", np.max(bfs))
    print("")


plt.figure(figsize=[4, 3])

fitness_traces("../T1", "xkcd:tomato", "Inverted Pendulum")
fitness_traces("../T2", "xkcd:azure", "Cart-Pole")
fitness_traces("../T3", "xkcd:teal green", "Legged Walker")

#plt.legend()
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.tight_layout()
plt.savefig("figure_1.pdf")
plt.show()
