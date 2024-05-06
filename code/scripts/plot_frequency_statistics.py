import matplotlib
import numpy as np
import torch

from optexp.experiments.bigger_models.gpt2small_wt103 import (
    gpt2small_wt103_with_class_stats_long,
)
from optexp.experiments.imbalance import PTB_with_class_stats
from optexp.experiments.paper_figures import get_dir
from optexp.experiments.toy_models import balanced_x_perclass
from optexp.experiments.vision.barcoded import mnist_and_barcoded
from optexp.plotter.plot_utils import subsample
from optexp.plotter.style_figure import update_plt
from optexp.utils import split_classes_by_frequency


def load_data(problem=gpt2small_wt103_with_class_stats_long):
    if hasattr(problem, "sgd_dataset"):
        problem.dataset = problem.sgd_dataset

    dataset = problem.dataset

    if (
        problem == PTB_with_class_stats
        or problem == gpt2small_wt103_with_class_stats_long
    ):
        tr_load = dataset.load()[0]
        ys = []
        for _, y in tr_load:
            ys.append(y)
        targets_np = torch.cat(ys).cpu().numpy()
        grouping = split_classes_by_frequency(targets_np, n_splits=10)
    elif problem == mnist_and_barcoded or problem == balanced_x_perclass:
        tr_load, _, _, _, _ = dataset.load()
        targets_np = tr_load.targets.cpu().numpy()
        if problem == mnist_and_barcoded:
            mnist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            other = list(set(np.unique(targets_np)) - set(mnist))
            grouping = (
                np.array(mnist, dtype=np.int64),
                np.array(other, dtype=np.int64),
            )
        else:
            grouping = split_classes_by_frequency(targets_np, n_splits=11)
    else:
        raise ValueError()

    return targets_np, grouping


def settings(plt):
    update_plt(plt, rel_width=0.275, nrows=1, ncols=1, height_to_width_ratio=0.7)


def make_figure(fig, data):
    targets_np, grouping = data

    unique, counts = np.unique(targets_np, return_counts=True)

    sort_idx = np.flip(np.argsort(counts))
    counts = counts[sort_idx]
    unique = unique[sort_idx]

    reverse_map = np.zeros((np.max(unique) + 1,), dtype=np.int64)
    reverse_map[unique] = np.arange(0, len(unique))

    for i, group in enumerate(grouping):
        print()
        print(
            "Group",
            i,
            "Number of classes:",
            len(group),
            "Number of samples:",
            np.sum(counts[reverse_map[group]]),
        )
        if True:
            if len(group) < 5:
                for i in group:
                    print(i, counts[reverse_map[i]])
            else:
                print("Med sample/class:", np.median(counts[reverse_map[group]]))

    ax = fig.add_subplot(111)

    cmap = matplotlib.cm.get_cmap("viridis")
    for i, group in enumerate(grouping):
        ax.plot(
            subsample(reverse_map[group] + 1, NMAX=6000),
            subsample(counts[reverse_map[group]], NMAX=6000),
            ".",
            markersize=3,
            color=cmap(i / (len(grouping) - 1)),
        )

    ax.set_title("# samples/class")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Class index (sorted)")
    ax.set_ylabel("# samples")

    zoom = 1.5
    print(f"n = {len(targets_np)}, Vocab size: {len(np.unique(targets_np))}")
    if len(targets_np) == 109836288:  # WT103
        ax.set_xlim(1 / zoom, 1e5 * zoom)
        ax.set_ylim(1 / zoom, 1e7 * zoom)
        ax.set_yticks([10**0, 10**3, 10**6])
        ax.set_xticks([10**0, 10**2, 10**4])
    if len(targets_np) == 923648:  # PTB
        ax.set_xlim(1 / zoom, 1e4 * zoom)
        ax.set_ylim(1 / zoom, 1e5 * zoom)
        ax.set_xticks([10**0, 10**2, 10**4])
        ax.set_yticks([10**0, 10**2, 10**4])
    if len(targets_np) == 101200:  # mnist_and_barcoded
        ax.set_xlim(1 / zoom, 1e4 * zoom)
        ax.set_ylim(1 / zoom, 1e5 * zoom)
        ax.set_xticks([10**0, 10**2, 10**4])
        ax.set_yticks([10**0, 10**2, 10**4])
    if len(targets_np) == 22528:  # balanced_x_perclass
        ax.set_xlim(1 / zoom, 1e4 * zoom)
        ax.set_ylim(1 / zoom, 1e4 * zoom)
        ax.set_xticks([10**0, 10**2, 10**4])
        ax.set_yticks([10**0, 10**2, 10**4])

    print(len(targets_np))

    fig.tight_layout(pad=0.1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    problems = [
        gpt2small_wt103_with_class_stats_long,
        PTB_with_class_stats,
        mnist_and_barcoded,
        balanced_x_perclass,
    ]
    for problem in problems:
        settings(plt)
        fig = plt.figure()
        make_figure(fig, load_data(problem=problem))
        fig.savefig(get_dir("histogram") / f"{problem.__name__}.pdf")
        plt.close(fig)
