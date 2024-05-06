import numpy as np
import torch
from tqdm import tqdm

from optexp.experiments.paper_figures import get_dir
from optexp.experiments.toy_models.balanced_x_perclass import optimizers, problem
from optexp.plotter.caching_helper import (
    hessian_subset_cache,
    model_cache,
    x_cache,
    y_cache,
)
from optexp.plotter.plot_utils import subsample
from optexp.plotter.style_figure import update_plt

sgd, sgdm, adam, adamm, nsgd, nsgdm, sign, signm = optimizers[:8]

# EPOCHS_TO_SAVE = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)
EPOCHS_TO_SAVE = (0, 10, 50, 100)


def process_hessian_subset(classes, dimensions, type, num):
    print("Starting process")
    for opt in tqdm([sgd, adamm]):
        w_done = all(
            [
                hessian_subset_cache.exists(type, num, opt, f"T={epoch}")
                for epoch in EPOCHS_TO_SAVE
            ]
        )
        if w_done:
            continue

        X, y = (x_cache.load(opt), y_cache.load(opt))
        for epoch in tqdm(EPOCHS_TO_SAVE):
            if hessian_subset_cache.exists(type, num, opt, f"T={epoch}"):
                continue

            problem.init_problem()
            problem.torch_model.load_state_dict(model_cache.load(opt, f"T={epoch}"))
            model = problem.torch_model.model
            model.to(X.device)

            probs = torch.nn.functional.softmax(model(X), dim=1)

            c_, d_ = len(classes), len(dimensions)
            diag_hessian_w = torch.zeros(c_, c_, d_, d_)

            Z = X[:, dimensions]

            for i, c1 in tqdm(enumerate(classes)):
                for j, c2 in enumerate(classes):
                    if c1 == c2:
                        weights = probs[:, c1] * (1 - probs[:, c1])
                    else:
                        weights = -probs[:, c1] * probs[:, c2]

                    diag_hessian_w[i, j, :, :] += torch.einsum(
                        "n,ni,nj->ij", weights, Z, Z
                    )

            hessian_subset_cache.save(
                diag_hessian_w,
                type,
                num,
                opt,
                f"T={epoch}",
            )


def subsample_cd(X, y, type, num):
    d = X.shape[1]
    c = len(torch.unique(y))
    dimensions = torch.tensor(
        subsample(np.array(range(d)), 2 * num, linear_only=True), dtype=torch.long
    )
    classes = None
    if type == "log":
        classes = torch.tensor(
            subsample(np.array(range(c)), 2 * num, log_only=True), dtype=torch.long
        )
    if type == "lin":
        classes = torch.tensor(
            subsample(np.array(range(c)), 2 * num, linear_only=True), dtype=torch.long
        )
    if type == "top":
        classes = torch.tensor(subsample(np.array(range(c))[:num]), dtype=torch.long)
    if type == "bot":
        classes = torch.tensor(subsample(np.array(range(c))[-num:]), dtype=torch.long)

    return classes, dimensions


# num = 10
# type = "top"
# type = "bot"
# type = "lin"
# type = "log"


def load_data(num=40, type="bot"):
    X, y = (x_cache.load(sgd), y_cache.load(sgd))
    classes, dimensions = subsample_cd(X, y, type, num)
    process_hessian_subset(classes, dimensions, type, num)
    ggns_adamm = [
        hessian_subset_cache.load(type, num, adamm, f"T={epoch}")
        for epoch in EPOCHS_TO_SAVE
    ]
    ggns_sgd = [
        hessian_subset_cache.load(type, num, sgd, f"T={epoch}")
        for epoch in EPOCHS_TO_SAVE
    ]

    return ggns_adamm, ggns_sgd, classes, dimensions, num, type, True


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=4, height_to_width_ratio=1.05)


def make_figure(fig, data):
    ggns_adamm, ggns_sgd, classes, dimensions, num, type, _ = data

    fig.set_dpi(100)

    axes = [
        fig.add_subplot(1, len(EPOCHS_TO_SAVE), 1 + i)
        for i in range(len(EPOCHS_TO_SAVE))
    ]

    c, d = len(classes), len(dimensions)

    def reshape_hessian(from_tensor):
        reshaped = np.zeros((c * d, c * d))
        for i1, c1 in enumerate(classes):
            for j1, c2 in enumerate(classes):
                reshaped[i1 * d : (i1 + 1) * d, j1 * d : (j1 + 1) * d] = from_tensor[
                    i1, j1, :, :
                ]
        return reshaped

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for i, epoch in enumerate(EPOCHS_TO_SAVE):
        H = reshape_hessian(ggns_adamm[i])
        H = np.log10(np.abs(H))
        show = axes[i].matshow(H, cmap="YlOrBr")
        cax = make_axes_locatable(axes[i]).append_axes("right", size="5%", pad=0.05)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis("off")
        fig.colorbar(show, cax=cax, orientation="vertical")
        axes[i].set_title("At epoch " + str(epoch))

    fig.tight_layout()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for type in ["top", "bot", "top", "lin", "log"]:
        data = load_data(10, type)
        fig = plt.figure()
        settings(plt)
        make_figure(fig, data)
        fig.savefig(get_dir("data/hess_sub") / f"{type}_{10}_subs_hess.pdf")
        fig.savefig(get_dir("data/hess_sub") / f"{type}_{10}_subs_hess.png", dpi=600)
        plt.close()
