import os
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


def dist(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))


def categorize(ipp, cpp, lwp, threshold):
    count = np.zeros(8)
    for (ip_neuron, cp_neuron, lw_neuron) in zip(ipp, cpp, lwp):
        if (
            ip_neuron > threshold and cp_neuron > threshold and lw_neuron > threshold
        ):  # no task neurons
            count[0] += 1
        if (
            ip_neuron <= threshold and cp_neuron > threshold and lw_neuron > threshold
        ):  # ip task neurons
            count[1] += 1
        if (
            ip_neuron > threshold and cp_neuron <= threshold and lw_neuron > threshold
        ):  # cp task neurons
            count[2] += 1
        if (
            ip_neuron > threshold and cp_neuron > threshold and lw_neuron <= threshold
        ):  # lw task neurons
            count[3] += 1
        if (
            ip_neuron <= threshold and cp_neuron <= threshold and lw_neuron > threshold
        ):  # ip + cp task neurons
            count[4] += 1
        if (
            ip_neuron <= threshold and cp_neuron > threshold and lw_neuron <= threshold
        ):  # ip + lw task neurons
            count[5] += 1
        if (
            ip_neuron > threshold and cp_neuron <= threshold and lw_neuron <= threshold
        ):  # cp + lw task neurons
            count[6] += 1
        if (
            ip_neuron <= threshold and cp_neuron <= threshold and lw_neuron <= threshold
        ):  # all  task neurons
            count[7] += 1

    return [count[1] + count[2] + count[3], np.sum(count[4:])], count


def lesion_compares(threshold):
    fig = plt.figure(figsize=[4.0, 2])

    # prep
    dir = "../New"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()

    all_contrib_dists_var = []
    all_contrib_dists_mi = []
    bin_contrib_dists_var = []
    bin_contrib_dists_mi = []

    for i, file in enumerate(files):
        fits = np.load(file)
        # if np.prod(fits) > 0.8:
        if np.min(fits) > 0.85:
            ind = file.split("/")[-1].split(".")[-2].split("_")[-1]

            l_ip = np.load("../New/lesions_IP_" + str(ind) + ".npy")
            l_cp = np.load("../New/lesions_CP_" + str(ind) + ".npy")
            l_lw = np.load("../New/lesions_LW_" + str(ind) + ".npy")
            lesion_data = np.concatenate([l_ip, l_cp, l_lw])
            lesion_bin_categs, lesion_categs = categorize(l_ip, l_cp, l_lw, threshold)

            mi_ip = 1 - np.load("../New/NormMI_IP_" + str(ind) + ".npy")
            mi_cp = 1 - np.load("../New/NormMI_CP_" + str(ind) + ".npy")
            mi_lw = 1 - np.load("../New/NormMI_LW_" + str(ind) + ".npy")
            mi_data = np.concatenate([mi_ip, mi_cp, mi_lw])
            mi_bin_categs, mi_categs = categorize(mi_ip, mi_cp, mi_lw, threshold)

            v_ip = 1 - np.load("../New/NormVar_IP_" + str(ind) + ".npy")
            v_cp = 1 - np.load("../New/NormVar_CP_" + str(ind) + ".npy")
            v_lw = 1 - np.load("../New/NormVar_LW_" + str(ind) + ".npy")
            v_data = np.concatenate([v_ip, v_cp, v_lw])
            var_bin_categs, var_categs = categorize(v_ip, v_cp, v_lw, threshold)

            all_contrib_dists_var.append(dist(v_data, lesion_data))
            all_contrib_dists_mi.append(dist(mi_data, lesion_data))
            bin_contrib_dists_var.append(dist(var_bin_categs, lesion_bin_categs))
            bin_contrib_dists_mi.append(dist(mi_bin_categs, lesion_bin_categs))

    ax1 = fig.add_subplot(1, 2, 1, adjustable="box", aspect=1)
    ax1.plot([0, 3.5], [0, 3.5], "k", linewidth=0.7)
    ax1.scatter(all_contrib_dists_mi, all_contrib_dists_var, c="C6", alpha=0.8, s=30)
    plt.xlim([0.0, 3.5])
    plt.ylim([0.0, 3.5])
    plt.xticks(np.arange(4))
    plt.yticks(np.arange(4))
    # plt.title("Estimated specialization vs reuse", fontsize=11)
    plt.xlabel("MI vs Lesion")
    plt.ylabel("Var vs Lesion")

    ax2 = fig.add_subplot(1, 2, 2, adjustable="box", aspect=1)
    ax2.plot([-0.5, 13.5], [-0.5, 13.5], "k", linewidth=0.7)
    ax2.scatter(bin_contrib_dists_mi, bin_contrib_dists_var, c="C9", alpha=0.8, s=30)
    plt.xlim([-0.5, 13.5])
    plt.ylim([-0.5, 13.5])
    plt.xticks(np.arange(0, 14, 4))
    plt.yticks(np.arange(0, 14, 4))
    # plt.title("Estimated contribution", fontsize=11)
    plt.xlabel("MI vs Lesion")
    plt.ylabel("Var vs Lesion")

    plt.tight_layout()
    plt.savefig("figure_9.pdf")
    plt.show()


lesion_compares(0.85)
