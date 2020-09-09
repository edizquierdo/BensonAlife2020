##################################################
# figure 4
# perf-map and behavior trials of the best
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def behavior_viz(dir, run_num):
    plt.figure(figsize=[8, 4])

    ## Perf maps
    plt.subplot(231)
    perfmap_file = os.path.join(dir, "perfmap_IP_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 4.5, 9], [-2.5, 0, 2.5])
    plt.yticks([0, 4.5, 9], [1, 0, -1])
    # plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("IP \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    plt.subplot(232)
    perfmap_file = os.path.join(dir, "perfmap_CP_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 4.5, 9], [-0.05, 0, 0.05])
    plt.yticks([0, 4.5, 9], [0.05, 0, -0.05])
    # plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("CP \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    plt.subplot(233)
    perfmap_file = os.path.join(dir, "perfmap_LW_{}.npy".format(run_num))
    perfmap = np.load(perfmap_file)
    plt.imshow(perfmap, vmin=0.8, vmax=1)
    plt.xticks([0, 4.5, 9], [-0.5, 0, 0.5])
    plt.yticks([0, 4.5, 9], [1, 0, -1])
    plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\omega$")
    plt.title("LW \n fitness = {}".format(np.round(np.mean(perfmap), decimals=3)))

    ## Behaviors
    plt.subplot(234)
    IP_dat = np.load(os.path.join(dir, "theta_traces_IP_{}.npy".format(run_num)))
    for theta_trace in IP_dat:
        plt.plot(np.arange(0, 10, 0.05), theta_trace)
    plt.yticks(np.arange(-360, 361, 180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del IP_dat

    plt.subplot(235)
    CP_dat = np.load(os.path.join(dir, "theta_traces_CP_{}.npy".format(run_num)))[::15]
    for theta_trace in CP_dat:
        plt.plot(theta_trace)
    # plt.yticks(np.arange(-360,361,180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del CP_dat

    plt.subplot(236)
    LW_dat = np.load(os.path.join(dir, "theta_traces_LW_{}.npy".format(run_num)))
    for theta_trace in LW_dat:
        plt.plot(theta_trace)
    # plt.yticks(np.arange(-360,361,180))
    plt.box(None)
    plt.ylabel(r"$\theta$")
    plt.xlabel("Time")
    del LW_dat

    plt.tight_layout()
    plt.savefig("figure_4.pdf")
    plt.show()


behavior_viz("../New", 74)
