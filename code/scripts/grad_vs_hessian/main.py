from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from optexp import Optimizer, Problem
from optexp.experiments.paper_figures import get_dir
from optexp.experiments.toy_models.balanced_x_perclass import optimizers, problem
from optexp.plotter.caching_helper import (
    CacheFile,
    grad_b_cache,
    grad_w_cache,
    hessian_b_cache,
    hessian_w_cache,
    model_cache,
    x_cache,
    y_cache,
)
from optexp.plotter.plot_utils import normalize_y_axis, subsample
from optexp.plotter.style_figure import update_plt

sgd, sgdm, adam, adamm = optimizers[:8]

EPOCHS_TO_SAVE = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)


def timestr(T, flipped=False):
    return f"T={T}" + ("_flipped" if flipped else "")


def run_and_save_checkpoints():
    def run_with_opt(prob: Problem, opt: Optimizer, epochs_to_save):
        if all([model_cache.exists(opt, f"T={t}") for t in epochs_to_save]):
            return

        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        prob.init_problem()
        x_cache.save(prob.train_loader.data, opt)
        y_cache.save(prob.train_loader.data, opt)

        if 0 in epochs_to_save:
            model_cache.save(prob.torch_model.state_dict(), opt, f"T=0")

        torch_opt = opt.load(prob.torch_model)
        for t in tqdm(list(range(1, max(epochs_to_save) + 1))):
            prob.one_epoch(torch_opt)
            if t in epochs_to_save:
                model_cache.save(prob.torch_model.state_dict(), opt, f"T={t}")

    run_with_opt(problem, sgd, epochs_to_save=EPOCHS_TO_SAVE)
    run_with_opt(problem, adamm, epochs_to_save=EPOCHS_TO_SAVE)


def process_grads(flipped):
    for opt in tqdm([sgd, adamm]):
        w_done = all(
            [
                grad_w_cache.exists(opt, timestr(epoch, flipped))
                for epoch in EPOCHS_TO_SAVE
            ]
        )
        b_done = all(
            [
                grad_b_cache.exists(opt, timestr(epoch, flipped))
                for epoch in EPOCHS_TO_SAVE
            ]
        )
        if w_done and b_done:
            continue

        X, y = (x_cache.load(opt), y_cache.load(opt))

        for epoch in tqdm(EPOCHS_TO_SAVE):
            if (
                grad_b_cache.get_path(opt, timestr(epoch, flipped)).exists()
                and grad_w_cache.get_path(opt, timestr(epoch, flipped)).exists()
            ):
                continue

            problem.init_problem()
            problem.torch_model.load_state_dict(model_cache.load(opt, timestr(epoch)))

            if flipped:
                problem.torch_model.model.weight.data = (
                    -problem.torch_model.model.weight.data
                )
                problem.torch_model.model.bias.data = (
                    -problem.torch_model.model.bias.data
                )

            lossfunc = torch.nn.CrossEntropyLoss()
            loss = lossfunc(problem.torch_model.model(X), y)

            loss.backward()

            grad_b_cache.save(
                problem.torch_model.model.bias.grad.detach(),
                opt,
                timestr(epoch, flipped),
            )
            grad_w_cache.save(
                problem.torch_model.model.weight.grad.detach(),
                opt,
                timestr(epoch, flipped),
            )


def process_diag_ggn_direct(flipped):
    for opt in tqdm([sgd, adamm]):
        w_done = all(
            [
                hessian_w_cache.exists(opt, timestr(epoch, flipped))
                for epoch in EPOCHS_TO_SAVE
            ]
        )
        b_done = all(
            [
                hessian_b_cache.exists(opt, timestr(epoch, flipped))
                for epoch in EPOCHS_TO_SAVE
            ]
        )
        if w_done and b_done:
            continue

        X, y = (x_cache.load(opt), y_cache.load(opt))
        for epoch in tqdm(EPOCHS_TO_SAVE):
            w_and_b_done = (
                hessian_w_cache.exists(opt, timestr(epoch, flipped)),
                hessian_b_cache.exists(opt, timestr(epoch, flipped)),
            )
            if all(w_and_b_done):
                continue

            problem.init_problem()
            problem.torch_model.load_state_dict(model_cache.load(opt, timestr(epoch)))

            if flipped:
                problem.torch_model.model.weight.data = (
                    -problem.torch_model.model.weight.data
                )
                problem.torch_model.model.bias.data = (
                    -problem.torch_model.model.bias.data
                )

            model = problem.torch_model.model
            model.to(X.device)

            probs = torch.nn.functional.softmax(model(X), dim=1)

            with torch.no_grad():
                diag_hessian_w = torch.zeros_like(model.weight)
                diag_hessian_b = torch.zeros_like(model.bias)
                classes = probs.shape[1]

                for i in tqdm(list(range(classes))):
                    weight = probs[:, i] * (1 - probs[:, i])
                    diag_hessian_w[i, :] += torch.einsum("n,nd->d", weight, X)
                    diag_hessian_b[i] += torch.sum(weight)

                hessian_w_cache.save(
                    diag_hessian_w.detach(), opt, timestr(epoch, flipped)
                )
                hessian_b_cache.save(
                    diag_hessian_w.detach(), opt, timestr(epoch, flipped)
                )


def load_data(flipped=True):
    run_and_save_checkpoints()
    process_grads(flipped)
    process_diag_ggn_direct(flipped)

    grads_adamm = [
        torch.load(grad_w_cache.get_path(adamm, timestr(epoch, flipped)))
        for epoch in EPOCHS_TO_SAVE
    ]
    grads_sgd = [
        torch.load(grad_w_cache.get_path(sgd, timestr(epoch, flipped)))
        for epoch in EPOCHS_TO_SAVE
    ]
    ggns_adamm = [
        torch.load(hessian_w_cache.get_path(adamm, timestr(epoch, flipped)))
        for epoch in EPOCHS_TO_SAVE
    ]
    ggns_sgd = [
        torch.load(hessian_w_cache.get_path(sgd, timestr(epoch, flipped)))
        for epoch in EPOCHS_TO_SAVE
    ]
    plot_to_make = "sgd"
    plot_to_make = "adam"
    plot_to_make = "correlation"
    return grads_adamm, grads_sgd, ggns_adamm, ggns_sgd, plot_to_make


def settings_longplot(plt):
    update_plt(plt, rel_width=1.0, nrows=3, ncols=8, height_to_width_ratio=1.05)


def settings_shortplot(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=6, height_to_width_ratio=1.35)


def settings(plt):
    update_plt(plt, rel_width=1.0, nrows=1, ncols=6, height_to_width_ratio=1.35)


def make_figure(fig, data):
    grads_adamm, grads_sgd, ggns_adamm, ggns_sgd, what_to_plot = data

    # plot_sgd = False
    # plot_sgd = True
    # sqrt_grads = False
    sqrt_grads = True
    divide_hessian_by_N = True

    fig.set_dpi(200)

    N = 22528

    # EPOCHS_TO_SAVE = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)
    epochs_to_plot = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)
    epochs_to_plot = (0, 1, 2, 3, 5, 10, 20, 30, 50, 100)
    if what_to_plot == "correlation":
        epochs_to_plot = [0, 1, 5, 10, 50, 100]
    else:
        epochs_to_plot = (0, 1, 2, 5, 10, 20, 50, 100)
    # epochs_to_plot = (0, 2, 5, 20, 50, 100)
    # epochs_to_plot = (0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100)
    idx_to_plot = [EPOCHS_TO_SAVE.index(epoch) for epoch in epochs_to_plot]

    grads_adamm = [grads_adamm[i] for i in idx_to_plot]
    grads_sgd = [grads_sgd[i] for i in idx_to_plot]
    ggns_adamm = [ggns_adamm[i] for i in idx_to_plot]
    ggns_sgd = [ggns_sgd[i] for i in idx_to_plot]

    if divide_hessian_by_N:
        ggns_adamm = [ggns_adamm[i] / N for i in range(len(ggns_adamm))]
        ggns_sgd = [ggns_sgd[i] / N for i in range(len(ggns_sgd))]

    cmap = matplotlib.cm.get_cmap("viridis")

    if sqrt_grads:
        lims_grads = [10**-5, 10**1]
        ticks_grads = [10**-4, 10**-2, 10**0]
        lims_hess = [10**-7, 10**-1]
        ticks_hess = [10**-6, 10**-4, 10**-2]
    else:
        lims_grads = [10**-10, 10**2]
        lims_hess = [10**-8, 10**0]
        ticks_grads = [10**-9, 10**-4, 10**1]
        ticks_hess = [10**-7, 10**-4, 10**-1]

    def compute_norm(grads_val):
        grads_norm_squared = ((grads_val) ** 2).sum(axis=1)
        return np.sqrt(grads_norm_squared) if sqrt_grads else grads_norm_squared

    def plot_corr(ax, grad_vals, ggn_vals):
        for pow in range(11):
            idx_start = 2**pow - 1
            idx_end = 2 ** (pow + 1) - 1

            NMAX = 100
            ax.plot(
                subsample(grad_vals[idx_start:idx_end], NMAX=NMAX),
                subsample(ggn_vals[idx_start:idx_end], NMAX=NMAX),
                ".",
                markersize=2,
                color=cmap(pow / 10),
            )

    def plot_on(axtop, axmid, axbot, grad_vals, ggn_vals):
        for pow in range(11):
            idx_start = 2**pow - 1
            idx_end = 2 ** (pow + 1) - 1
            indices = np.array(list(range(idx_start, idx_end)))

            NMAX = 100
            axtop.plot(
                subsample(indices + 1, NMAX=NMAX),
                subsample(grad_vals[idx_start:idx_end], NMAX=NMAX),
                ".",
                markersize=2,
                color=cmap(pow / 10),
            )
            axmid.plot(
                subsample(indices + 1, NMAX=NMAX),
                subsample(ggn_vals[idx_start:idx_end], NMAX=NMAX),
                ".",
                markersize=2,
                color=cmap(pow / 10),
            )
        plot_corr(axbot, grad_vals, ggn_vals)

    def plot_single(ggns, grads, sqrt_grads):
        axes = [
            [fig.add_subplot(3, len(grads), i + 1) for i in range(len(grads))],
            [
                fig.add_subplot(3, len(grads), len(grads) + i + 1)
                for i in range(len(grads))
            ],
            [
                fig.add_subplot(3, len(grads), 2 * len(grads) + i + 1)
                for i in range(len(grads))
            ],
        ]
        for i in range(len(grads)):
            axtop = axes[0][i]
            axmid = axes[1][i]
            axbot = axes[-1][i]

            grad_vals = compute_norm(grads[i].detach().numpy())
            ggn_vals = ggns[i].detach().numpy().mean(axis=1)

            plot_on(axtop, axmid, axbot, grad_vals, ggn_vals)

            for ax in [axtop, axmid, axbot]:
                ax.set_yscale("log")
                ax.set_xscale("log")

            axtop.set_title("t=" + str(epochs_to_plot[i]))

            for ax in [axtop, axmid]:
                ax.set_xlim(1 / 2, axtop.get_xlim()[1] * 1.10)
                ax.set_xticks([1, 10, 100, 1000])
                ax.set_xticklabels([1, "", "", "10$^3$"])
        for ax in axes[0][1:] + axes[1][1:] + axes[-1][1:]:
            ax.set_yticklabels([])
        if sqrt_grads:
            axes[0][0].set_ylabel("Gradient\n" + r"$\Vert \nabla_{w_c} \, f(W_t)\Vert$")
        else:
            axes[0][0].set_ylabel(
                "Gradient\n" + r"$\Vert \nabla_{w_c} \, f(W_t)\Vert^2$"
            )
        axes[1][0].set_ylabel("Hessian\n" + r"Tr$(\nabla_{w_c}^2 \, f(W_t))$")
        axes[2][0].set_ylabel("Hessian")
        for ax in axes[1]:
            ax.set_xlabel("Class ", labelpad=-5)
        for ax in axes[-1]:
            ax.set_xlabel(" Grad", labelpad=-5)

        for ax in axes[0]:
            ax.set_ylim(lims_grads)
            ax.set_yticks(ticks_grads)
        for ax in axes[1]:
            ax.set_ylim(lims_hess)
            ax.set_yticks(ticks_hess)
        for ax in axes[2]:
            ax.set_xlim(lims_grads)
            ax.set_xticks(ticks_grads)
            ax.set_xticklabels(
                [
                    "$10^{" + str(int(np.log10(ticks_grads[0]))) + "}$",
                    "",
                    "$10^{" + str(int(np.log10(ticks_grads[-1]))) + "}$",
                ]
            )
            ax.set_ylim(lims_hess)
            ax.set_yticks(ticks_hess)

        normalize_y_axis(*axes[0])
        normalize_y_axis(*axes[1])
        # plt.subplot_tool()
        fig.tight_layout(pad=0.01)

    def plot_correlation():
        K = len(grads_sgd)
        axes = [
            [fig.add_subplot(1, K, i + 1) for i in range(K)],
        ]

        for i in range(K):
            plot_corr(
                axes[0][i],
                compute_norm(grads_adamm[i].detach().numpy()),
                ggns_adamm[i].detach().numpy().mean(axis=1),
            )
            axes[0][i].plot(
                [10**-5, 10**1],
                [10**-7, 10**-1],
                color="gray",
                alpha=0.5,
                linestyle="--",
                linewidth=1.0,
            )

        for i, t in enumerate(epochs_to_plot):
            if i == 0:
                axes[0][0].set_title("Correlation at \n $t=0$", y=0.825)
            else:
                axes[0][i].set_title("$t=" + str(t) + "$")

        for row in axes:
            for ax in row:
                ax.set_yscale("log")
                ax.set_xscale("log")

        for ax in axes[0]:
            ax.set_xlim(1 / 2, ax.get_xlim()[1] * 1.10)
            ax.set_xticks([1, 10, 100, 1000])
            ax.set_xticklabels([1, "", "", "10$^3$"])

        for ax in axes[0][1:]:
            ax.set_yticklabels([])

        axes[0][0].set_ylabel("Hessian\n" + r"Tr$(\nabla_{w_c}^2  \,f\,\,)$")
        for i in range(K):
            axes[0][i].set_xlabel(
                "Grad.\n" + r"$\Vert \nabla_{w_c} \, f\Vert$", labelpad=-7
            )

        for ax in axes[0]:
            ax.set_xlim(lims_grads)
            ax.set_xticks(ticks_grads)
            ax.set_xticklabels(
                [
                    "$10^{" + str(int(np.log10(ticks_grads[0]))) + "}$",
                    "",
                    "$10^{" + str(int(np.log10(ticks_grads[-1]))) + "}$",
                ]
            )
            ax.set_ylim(lims_hess)
            ax.set_yticks(ticks_hess)

        normalize_y_axis(*axes[0])

        fig.tight_layout(pad=0.5, rect=(0.0, 0, 1, 1.0))

    if what_to_plot == "correlation":
        plot_correlation()
    elif what_to_plot == "sgd":
        plot_single(ggns_sgd, grads_sgd, sqrt_grads)
    elif what_to_plot == "adam":
        plot_single(ggns_adamm, grads_adamm, sqrt_grads)
    else:
        raise ValueError(f"Unknown plot type {what_to_plot}")


if __name__ == "__main__":
    ##
    # The following functions can take a while
    #
    # Run the optimizers again and save the checkpoints
    # run_and_save_checkpoints() # ~2 hours
    # Load the checkpoints and save the gradients
    # process_grads() # ~20min
    # Load the checkpoints and save the diagonal hessian
    # process_diag_ggn_mc() # ~20min

    settings_shortplot(plt)
    data = load_data()
    grads_adamm, grads_sgd, ggns_adamm, ggns_sgd, plot_sgd = data

    fig = plt.figure()
    make_figure(
        fig,
        (
            grads_adamm,
            grads_sgd,
            ggns_adamm,
            ggns_sgd,
            "correlation",
        ),
    )
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_corr.pdf")
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_corr.png", dpi=600)
    plt.close(fig)

    settings_longplot(plt)
    fig = plt.figure()
    make_figure(
        fig,
        (
            grads_adamm,
            grads_sgd,
            ggns_adamm,
            ggns_sgd,
            "sgd",
        ),
    )
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_sgd.pdf")
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_sgd.png", dpi=600)
    plt.close(fig)

    fig = plt.figure()
    make_figure(
        fig,
        (
            grads_adamm,
            grads_sgd,
            ggns_adamm,
            ggns_sgd,
            "adam",
        ),
    )
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_adam.pdf")
    fig.savefig(get_dir("grad_hessian") / "gradient_hessian_adam.png", dpi=600)
    plt.close(fig)
