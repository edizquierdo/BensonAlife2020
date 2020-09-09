##################################################
# figure 6
# comparing lesion, mi and var analysis
##################################################
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


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


def compare_analyses(threshold):
    plt.figure(figsize=[8, 3])
    # prep
    plt.subplot(131)
    plt.plot([0, 3.5], [0, 3.5], "k", linewidth=0.7)
    plt.subplot(132)
    plt.plot([0, 13.5], [0, 13.5], "k", linewidth=0.7)
    plt.subplot(133)
    plt.plot([0, 12], [0, 12], "k", linewidth=0.7)

    dir = "../New"
    files = glob.glob(os.path.join(dir, "perf_*.npy"))
    files.sort()

    all_categs = []
    all_counts = []

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

            plt.subplot(131)
            plt.scatter(
                dist(mi_data, lesion_data),
                dist(v_data, lesion_data),
                c="k" if ind == "74" else "C0",
                s=30,
                alpha=1 if ind == "74" else 0.8,
            )

            plt.subplot(132)
            plt.scatter(
                dist(mi_bin_categs, lesion_bin_categs),
                dist(var_bin_categs, lesion_bin_categs),
                c="k" if ind == "74" else "C1",
                s=30,
                alpha=1 if ind == "74" else 0.8,
            )

            plt.subplot(133)
            plt.scatter(
                dist(mi_categs, lesion_categs),
                dist(var_categs, lesion_categs),
                c="k" if ind == "74" else "C2",
                s=30,
                alpha=1 if ind == "74" else 0.8,
            )

    plt.subplot(131, aspect=1)
    plt.xlim([0, 3.5])
    plt.ylim([0, 3.5])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated contribution\nMI vs Lesion")
    plt.ylabel("Estimated contribution\nNeural variability vs Lesion")

    plt.subplot(132, aspect=1)
    plt.xlim([0, 13.5])
    plt.ylim([0, 13.5])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated specialization vs reuse\nMI vs Lesion")
    plt.ylabel("Estimated specialization vs reuse\nNeural variability vs Lesion")

    plt.subplot(133, aspect=1)
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    plt.axes(aspect="equal")
    plt.xlabel("Estimated reuse categories\nMI vs Lesion")
    plt.ylabel("Estimated reuse categories\nNeural variability vs Lesion")

    plt.tight_layout()
    plt.savefig("figure_6c.pdf")
    plt.savefig("figure_6c.png")
    plt.show()


compare_analyses(0.85)
