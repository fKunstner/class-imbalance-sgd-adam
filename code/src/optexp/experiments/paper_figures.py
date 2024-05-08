import os

from optexp import SGD, Adam, Experiment, NormSGD, config
from optexp.experiments.bigger_models.gpt2small_wt103 import (
    gpt2small_wt103_with_class_stats_long,
)
from optexp.experiments.imbalance import (
    PTB_class_weighted_per_class,
    PTB_with_class_stats,
)
from optexp.experiments.simpler_transformers import (
    basic_one_layer_perclass,
    train_only_last_layer_perclass,
)
from optexp.experiments.toy_models import balanced_x_perclass
from optexp.experiments.vision import mnist_only
from optexp.experiments.vision.barcoded import (
    mnist_and_barcoded_long_perclass,
    mnist_and_barcoded_perclass,
    mnist_barcoded_only_long,
)
from optexp.optimizers.normalized_opts import Sign
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.plot_per_class import plot_per_class


def get_dir(save_dir: str):
    dir = config.get_plots_directory() / "paper" / save_dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def select_nomom(exp: Experiment):
    if isinstance(exp.optim, Adam):
        if exp.optim.beta1 == 0.0:
            return True
    else:
        assert (
            isinstance(exp.optim, SGD)
            or isinstance(exp.optim, NormSGD)
            or isinstance(exp.optim, Sign)
        )
        if exp.optim.momentum == 0.0:
            return True


def select_SGDM_and_AdamM(exp: Experiment):
    if isinstance(exp.optim, SGD):
        if exp.optim.momentum > 0.0:
            return True
    elif isinstance(exp.optim, Adam):
        if exp.optim.beta1 > 0.0:
            return True
    return False


def select_extended_optimizers(exp: Experiment):
    return (
        isinstance(exp.optim, SGD)
        or isinstance(exp.optim, Adam)
        or isinstance(exp.optim, NormSGD)
        or isinstance(exp.optim, Sign)
    )


def select_seed_0(exp: Experiment):
    return exp.seed == 0


H_TO_W_RATIO = 0.8
H_TO_W_RATIO_1_PLOT = 0.7
WIDTH_5_PLOTS = 1.0
WIDTH_3_PLOTS = 0.72
WIDTH_1_PLOT = 0.275


def fig1():
    experiments = gpt2small_wt103_with_class_stats_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig1"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
        using_step=True,
    )


def fig2():
    experiments = balanced_x_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig2"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig2_shorter_timescale():
    experiments = balanced_x_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        plotting_time=150,
        where=get_dir("fig2_shorter_timescale"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig2_barcode_feasable():
    experiments = mnist_barcoded_only_long.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_best(
        experiments=experiments,
        where=get_dir("fig2_barcode_feasable"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=400,
    )


def fig2_additional_opts():
    experiments = mnist_and_barcoded_long_perclass.experiments
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig2/additional_opts_perclass"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=400,
    )


def fig3():
    experiments = mnist_only.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(experiments)
    plot_best(
        experiments=experiments,
        where=get_dir("fig3/balanced"),
        rel_width=WIDTH_1_PLOT,
        height_to_width_ratio=H_TO_W_RATIO_1_PLOT,
    )

    experiments = mnist_and_barcoded_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = filter(select_seed_0, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig3/imbalanced"),
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig4(epoch=1000):
    experiments = train_only_last_layer_perclass.experiments
    experiments = filter(select_nomom, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/nomom_{epoch}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epoch,
    )

    experiments = train_only_last_layer_perclass.experiments
    experiments = filter(lambda x: not select_nomom(x), experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/mom_{epoch}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epoch,
    )


def fig_linear_more_opts(epochs=500):
    experiments = balanced_x_perclass.experiments
    experiments = filter(select_extended_optimizers, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/linear_more_opts_{epochs}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epochs,
    )


def fig_ptb_class_stats(epochs=500):
    experiments = basic_one_layer_perclass.experiments
    experiments = filter(select_extended_optimizers, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"fig4/ptb_more_opts_{epochs}"),
        rel_width=1.0,
        height_to_width_ratio=H_TO_W_RATIO,
        plotting_time=epochs,
    )


def fig7():
    experiments = PTB_class_weighted_per_class.experiments
    # experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir("fig7"),
        rel_width=WIDTH_5_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig_appendix_PTB_stochastic(epochs=100):
    experiments = PTB_with_class_stats.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"appendix_standard_training_{epochs}"),
        plotting_time=epochs,
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


def fig_appendix_PTB_deterministic(epochs=500):
    experiments = basic_one_layer_perclass.experiments
    experiments = filter(select_SGDM_and_AdamM, experiments)
    experiments = list(experiments)
    plot_per_class(
        experiments=experiments,
        where=get_dir(f"appendix_fb_training_{epochs}"),
        plotting_time=epochs,
        rel_width=WIDTH_3_PLOTS,
        height_to_width_ratio=H_TO_W_RATIO,
    )


if __name__ == "__main__":

    # Big transformer
    fig1()

    # It also works on smaller models
    fig_appendix_PTB_stochastic(epochs=100)
    fig_appendix_PTB_stochastic(epochs=50)

    # And in deterministic setting
    fig_appendix_PTB_deterministic(epochs=500)
    fig_appendix_PTB_deterministic(epochs=200)

    # And on vision
    fig2()
    fig2_barcode_feasable()
    fig2_shorter_timescale()
    fig2_additional_opts()

    # And on linear models
    fig3()

    # And with other optimizers
    fig4(1000)
    fig4(500)
    fig4(200)

    # More other optimizers
    fig_linear_more_opts(500)
    fig_linear_more_opts(200)
    fig_ptb_class_stats(500)
    fig_ptb_class_stats(200)

    # Reweighting
    fig7()
