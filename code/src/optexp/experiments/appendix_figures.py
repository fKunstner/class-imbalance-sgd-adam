from dataclasses import dataclass
from fractions import Fraction
from typing import Type

from optexp import SGD, Adam, Optimizer
from optexp.experiments.paper_figures import (
    H_TO_W_RATIO,
    WIDTH_1_PLOT,
    WIDTH_3_PLOTS,
    get_dir,
    select_nomom,
    select_seed_0,
)
from optexp.experiments.toy_models.smaller import (
    balanced_x_smaller_longer,
    balanced_x_smaller_longer_perclass,
)
from optexp.experiments.toy_models.smaller.changing_input import nnz_mean, zero_mean
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.plot_per_class import plot_per_class


def select_sgd_with_lr(lr_exponent: Fraction):
    def select_sgd_with_lr_inner(exp):
        if isinstance(exp.optim, SGD):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_sgd_with_lr_inner


def select_adam_with_lr(lr_exponent: Fraction):
    def select_adam_with_lr_inner(exp):
        if isinstance(exp.optim, Adam):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_adam_with_lr_inner


def selector_lr(lr_exponent: Fraction):
    def select_lr(exp):
        if isinstance(exp.optim, Optimizer):
            if exp.optim.learning_rate.exponent == lr_exponent:
                return True
        return False

    return select_lr


def fig_large_ss():
    experiments = balanced_x_smaller_longer.experiments
    experiments = filter(select_nomom, experiments)
    experiments = list(filter(select_seed_0, experiments))
    experiments_large = list(filter(select_sgd_with_lr(Fraction(3, 1)), experiments))
    experiments_small = list(filter(select_sgd_with_lr(Fraction(-1, 1)), experiments))

    def postprocess(fig):
        for ax in fig.get_axes():
            ax.set_title("Large step-size ($10^{3}$)")
            ax.set_ylim([10**-2, 10**5.5])
        fig.tight_layout(pad=0)

    plot_best(
        experiments=experiments_large,
        where=get_dir("additional/longer/large_ss"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=0.8,
        plotting_time=1000,
        postprocess=postprocess,
    )

    def postprocess(fig):
        for ax in fig.get_axes():
            ax.set_title("Small step-size ($10^{-1}$)")
            ax.set_ylim([10**-2, 10**5.5])
        fig.tight_layout(pad=0)

    plot_best(
        experiments=experiments_small,
        where=get_dir("additional/longer/small_ss"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=0.8,
        plotting_time=1000,
        postprocess=postprocess,
    )


def fig_longterm():
    experiments = balanced_x_smaller_longer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = filter(select_sgd_with_lr(Fraction(-2, 1)), experiments)
    experiments = list(experiments)

    times = [100, 1000, 10000]
    titles = ["Short (100 steps)", "Medium (1k steps)", "Long (10k steps)"]
    for time, title in zip(times, titles):

        def postprocess(fig):
            for ax in fig.get_axes():
                if ax.get_yscale() != "log":
                    ax.set_ylim([0, 7])
                ax.set_title(title)
            fig.tight_layout(pad=0)

        plot_per_class(
            experiments=experiments,
            where=get_dir(f"additional/longer/longer_run_{time}"),
            rel_width=WIDTH_1_PLOT,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=time,
            postprocess=postprocess,
            plot_overall_loss=False,
        )


def fig_compare_stepsizes(optimizer: Type[SGD | Adam] = SGD, start_at=-3):
    experiments = balanced_x_smaller_longer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(filter(lambda _: isinstance(_.optim, optimizer), experiments))

    experiments = (
        list(filter(selector_lr(Fraction(start_at, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 1, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 2, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 3, 2)), experiments))
        + list(filter(selector_lr(Fraction(start_at + 4, 2)), experiments))
    )

    name = "SGD" if optimizer == SGD else "Adam"
    stopping_times = [10000, 3162, 1000, 316, 100]

    for exp, stopping_time in zip(experiments, stopping_times):

        def postprocess(fig):
            for ax in fig.get_axes():
                if ax.get_yscale() != "log":
                    ax.set_ylim([0, 7])
                ax.set_title(
                    f"{name}, $\\alpha$ = {exp.optim.learning_rate.as_latex_str()}"
                )
            fig.tight_layout(pad=0)

        plot_per_class(
            experiments=[exp],
            where=get_dir(
                f"additional/per_ss/start_at_{start_at}/{name}/{exp.optim.learning_rate}"
            ),
            rel_width=WIDTH_1_PLOT,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=stopping_time,
            postprocess=postprocess,
            plot_overall_loss=False,
        )


def compare_all_stepsizes():
    for start_at in [-8, -7, -6, -5, -4]:
        fig_compare_stepsizes(optimizer=SGD, start_at=start_at)
    for start_at in [-12, -11, -10, -9, -8, -7, -6]:
        fig_compare_stepsizes(optimizer=Adam, start_at=start_at)


def fig_compare_inputs():

    exps_nnz = nnz_mean.experiments
    exps_nnz = filter(select_seed_0, exps_nnz)
    exps_nnz = list(filter(select_nomom, exps_nnz))
    exps_nnz = list(filter(select_sgd_with_lr(Fraction(-4, 2)), exps_nnz)) + list(
        filter(select_adam_with_lr(Fraction(-6, 2)), exps_nnz)
    )
    exps_zero = zero_mean.experiments
    exps_zero = filter(select_seed_0, exps_zero)
    exps_zero = list(filter(select_nomom, exps_zero))
    exps_zero = list(filter(select_sgd_with_lr(Fraction(4, 2)), exps_zero)) + list(
        filter(select_adam_with_lr(Fraction(-3, 2)), exps_zero)
    )

    def postprocess(fig):
        for ax in fig.get_axes():
            if ax.get_yscale() != "log":
                ax.set_ylim([0, 7])
            # ax.set_title("")
        fig.tight_layout(pad=0.3)

    for name, exp in zip(["nnz", "zero"], [exps_nnz, exps_zero]):
        plot_per_class(
            experiments=exp,
            where=get_dir(f"additional/compare_inputs/{name}"),
            rel_width=WIDTH_3_PLOTS,
            height_to_width_ratio=H_TO_W_RATIO,
            plotting_time=25,
            postprocess=postprocess,
            plot_overall_loss=True,
        )


if __name__ == "__main__":
    # fig_large_ss()
    # fig_longterm()
    # fig_compare_all_stepsizes()
    fig_compare_inputs()
