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


def contrib_best(file_name, run_num, label):
    plt.plot([0.5, 10.5], [0.15, 0.15], "k--", alpha=0.7)
    for i in np.arange(1, 10):
        plt.plot([i + 0.5, i + 0.5], [-0.05, 1.1], "gray", alpha=0.5)

    # load and plot data
    contrib_data = np.load("../New/{}_IP_{}.npy".format(file_name, run_num))
    plt.scatter(
        np.arange(1, 11) - 0.05,
        contrib_data,
        s=50,
        alpha=0.7,
        c="xkcd:tomato",
        label="IP",
    )
    contrib_data = np.load("../New/{}_CP_{}.npy".format(file_name, run_num))
    plt.scatter(
        np.arange(1, 11), contrib_data, s=50, alpha=0.7, c="xkcd:azure", label="CP"
    )
    contrib_data = np.load("../New/{}_LW_{}.npy".format(file_name, run_num))
    plt.scatter(
        np.arange(1, 11) + 0.05,
        contrib_data,
        s=50,
        alpha=0.7,
        c="xkcd:teal green",
        label="LW",
    )

    plt.xticks(np.arange(1, 11))
    plt.xlim([0.5, 10.5])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Neuron #")
    plt.ylabel(label)


def lesion_compares(threshold):
    fig = plt.figure(figsize=[8, 5])

    plt.subplot(231)
    contrib_best("NormVar", 74, "Normalized\nneural variability")

    plt.subplot(234)
    contrib_best("NormMI", 74, "Normalized\nmutual information")

    # prep
    dir = "../New"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()

    all_mi_bin_categs = []
    all_var_bin_categs = []

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
            all_mi_bin_categs.append(mi_bin_categs)

            v_ip = 1 - np.load("../New/NormVar_IP_" + str(ind) + ".npy")
            v_cp = 1 - np.load("../New/NormVar_CP_" + str(ind) + ".npy")
            v_lw = 1 - np.load("../New/NormVar_LW_" + str(ind) + ".npy")
            v_data = np.concatenate([v_ip, v_cp, v_lw])
            var_bin_categs, var_categs = categorize(v_ip, v_cp, v_lw, threshold)
            all_var_bin_categs.append(var_bin_categs)

    # plt.subplot(233)
    ax1 = fig.add_subplot(2, 3, 3, adjustable="box", aspect=1)
    ax1.plot([-0.5, 10.25], [10.25, -0.5], "k", linewidth=0.7)
    df = pd.DataFrame(
        all_var_bin_categs,
        columns=["No. of specialized neurons", "No. of reused neurons"],
    )
    ax = sns.swarmplot(
        x="No. of specialized neurons",
        y="No. of reused neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(11)]),
        alpha=0.8,
        s=7,
    )
    plt.xticks(np.arange(11), np.arange(11))
    plt.yticks(np.arange(11), np.arange(11))
    plt.xlim([-0.5, 10.25])
    plt.ylim([-0.5, 10.25])

    # plt.subplot(236)
    ax2 = fig.add_subplot(2, 3, 6, adjustable="box", aspect=1)
    ax2.plot([-0.5, 10.25], [10.25, -0.5], "k", linewidth=0.7)
    df = pd.DataFrame(
        all_mi_bin_categs,
        columns=["No. of specialized neurons", "No. of reused neurons"],
    )
    ax = sns.swarmplot(
        x="No. of specialized neurons",
        y="No. of reused neurons",
        data=df,
        palette=dict([(i, "xkcd:velvet") for i in range(11)]),
        alpha=0.8,
        s=7,
    )
    plt.xticks(np.arange(11), np.arange(11))
    plt.yticks(np.arange(11), np.arange(11))
    plt.xlim([-0.5, 10.25])
    plt.ylim([-0.5, 10.25])

    plt.tight_layout()
    plt.savefig("figure_678.pdf")
    plt.show()


lesion_compares(0.85)
