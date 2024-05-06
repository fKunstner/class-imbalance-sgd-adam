import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from optexp.experiments.paper_figures import get_dir
from optexp.experiments.toy_models.balanced_x_perclass import optimizers, problem
from optexp.plotter.caching_helper import model_cache, probs_cache, x_cache, y_cache
from optexp.plotter.style_figure import update_plt

sgd, sgdm, adam, adamm, nsgd, nsgdm, sign, signm = optimizers[:8]
EPOCHS_TO_SAVE = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)


def process_probs():
    for opt in tqdm([sgd, adamm]):
        probs_done = all(
            [probs_cache.exists(opt, f"T={epoch}") for epoch in EPOCHS_TO_SAVE]
        )
        import pdb

        pdb.set_trace()
        if probs_done:
            continue

        X, y = (x_cache.load(opt), y_cache.load(opt))
        for epoch in tqdm(EPOCHS_TO_SAVE):
            if probs_cache.exists(opt, epoch):
                continue

            problem.init_problem()
            problem.torch_model.load_state_dict(model_cache.load(opt, f"T={epoch}"))
            X.to("cpu")
            problem.torch_model.to("cpu")
            probs = problem.torch_model(X)

            probs_cache.save(probs, opt, epoch)


def load_data():
    process_probs()
    logits_adam = [probs_cache.load(adamm, f"T={epoch}") for epoch in EPOCHS_TO_SAVE]
    logits_sgd = [probs_cache.load(sgd, f"T={epoch}") for epoch in EPOCHS_TO_SAVE]
    X_adam, y_adam = (x_cache.load(adamm), y_cache.load(adamm))
    X_sgd, y_sgd = (x_cache.load(sgd), y_cache.load(sgd))
    return (X_adam, y_adam, X_sgd, y_sgd, logits_adam, logits_sgd, True)


def postprocess(data):
    X_adam, y_adam, X_sgd, y_sgd, logits_adam, logits_sgd, _ = data

    logits_adam = list([_.detach() for _ in logits_adam])
    logits_sgd = list([_.detach() for _ in logits_sgd])

    print(len(logits_adam))
    print(logits_adam[0].shape)

    ys = list(y_adam)

    unique_classes, classes_counts = np.unique(ys, return_counts=True)
    unique_classes, classes_counts = unique_classes.tolist(), classes_counts.tolist()
    unique_class_counts = np.unique(classes_counts)
    unique_class_counts = unique_class_counts.tolist()

    groups = [[] for _ in unique_class_counts]
    for class_id, class_count in zip(unique_classes, classes_counts):
        groups[unique_class_counts.index(class_count)].append(class_id)

    timesteps = len(logits_adam)
    n = len(y_adam)
    c = len(unique_classes)
    g = len(groups)
    data = [[[] for _ in groups] for _ in range(timesteps)]

    def split_timestep(ps):
        data = [[] for _ in groups]
        for i in range(n):
            class_for_sample = y_adam[i]
            group_idx = unique_class_counts.index(classes_counts[class_for_sample])
            data[group_idx].append(ps[i, class_for_sample])
        return data

    data_adam = [
        split_timestep(torch.nn.functional.softmax(logits_adam[idx], dim=1))
        for idx in range(timesteps)
    ]
    data_sgd = [
        split_timestep(torch.nn.functional.softmax(logits_sgd[idx], dim=1))
        for idx in range(timesteps)
    ]

    return data_adam, data_sgd


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=2, height_to_width_ratio=1.05)


def make_figure(fig, data):
    ps_adam, ps_sgd = data

    fig.set_dpi(100)

    epochs_to_plot = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)
    idx_to_plot = list([EPOCHS_TO_SAVE.index(epoch) for epoch in epochs_to_plot])

    N = len(idx_to_plot)
    cmap = matplotlib.cm.get_cmap("viridis")

    axes = [fig.add_subplot(1, 2, 1 + i) for i in range(2)]

    data_to_plot = ps_sgd

    def plot_data_on_axes(ax, data_to_plot):

        for group in range(len(data_to_plot[0])):
            ax.plot(
                [_ + 1 for _ in epochs_to_plot],
                [np.median(data_to_plot[idx][group]) for idx in idx_to_plot],
                ".-",
                label="Adam",
                color=cmap(1 - group / (len(data_to_plot[0]) - 1)),
            )
            for q in [95]:
                ax.fill_between(
                    [_ + 1 for _ in epochs_to_plot],
                    [
                        np.quantile(data_to_plot[idx][group], q=(100 - q) / 100)
                        for idx in idx_to_plot
                    ],
                    [
                        np.quantile(data_to_plot[idx][group], q=q / 100)
                        for idx in idx_to_plot
                    ],
                    alpha=0.2,
                    facecolor="none",
                    linewidth=0.0,
                    color=cmap(1 - group / (len(data_to_plot[0]) - 1)),
                )

        ax.axhline(0.5, linestyle="--", color="k")
        ax.set_ylim([axes[0].get_ylim()[0], 1.0])
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.set_xscale("log")

    plot_data_on_axes(axes[0], ps_adam)
    plot_data_on_axes(axes[1], ps_sgd)

    axes[0].set_title("Adam")
    axes[1].set_title("GD")
    axes[0].set_ylabel("Predicted Probability for correct class\n(split by group)")

    fig.tight_layout(pad=0.5)


if __name__ == "__main__":
    settings(plt)
    data = load_data()
    data = postprocess(data)
    fig = plt.figure()
    make_figure(fig, data)

    fig.savefig(get_dir("data/hess_sub") / "probs_over_time.pdf")
    fig.savefig(get_dir("data/hess_sub") / "probs_over_time.png", dpi=600)
