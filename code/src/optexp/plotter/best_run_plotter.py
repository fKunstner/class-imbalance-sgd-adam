import hashlib
import os
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from optexp import Experiment, LightningExperiment, config
from optexp.plotter.data_utils import (
    MEASURES,
    get_best,
    get_exps_data_epoch,
    load_data_for_exps,
)
from optexp.plotter.names_and_consts import (
    displayname,
    get_ylims_for_problem_linear,
    metrics_to_plot_and_main_metric_for_standard_plots,
    should_plot_logy,
)
from optexp.plotter.plot_utils import subsample
from optexp.plotter.style_figure import make_fig_axs
from optexp.plotter.style_lines import PlotOptimizer, get_optimizers


def plot_best(
    experiments: List[Experiment],
    plotting_time: Optional[int] = None,
    using_step: Optional[bool] = False,
    where: Optional[Path] = None,
    rel_width: float = 0.5,
    height_to_width_ratio=1.0,
    postprocess=None,
):
    group = experiments[0].group
    problem = experiments[0].problem

    if using_step and isinstance(experiments[0], LightningExperiment):
        eval_every = experiments[0].eval_every
        plotting_time = plotting_time if plotting_time else experiments[0].steps
    else:
        eval_every = 1
        plotting_time = plotting_time if plotting_time else experiments[0].epochs

    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")

    if using_step:
        all_lightning = all(isinstance(exp, LightningExperiment) for exp in experiments)
        same_have_eval_every = all(exp.eval_every == eval_every for exp in experiments)
        if not all_lightning:
            raise ValueError("All experiments must be Lightning experiments.")
        if not same_have_eval_every:
            warnings.warn(
                "Not all experiments have the same eval_every. Something might break"
            )

    if where is None:
        base_save_dir = config.get_plots_directory() / Path(group) / Path("best")
    else:
        base_save_dir = where

    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)

    main_metric, metric_names = metrics_to_plot_and_main_metric_for_standard_plots(
        problem
    )

    if plotting_time % eval_every != 0:
        raise ValueError(
            "Evaluation to plot up to must be a multiple of the evaluation interval"
        )

    exps_w_data = load_data_for_exps(experiments)
    exps_df = get_exps_data_epoch(exps_w_data, metric_names, plotting_time, using_step)

    optimizers = get_optimizers()
    for opt_plot in optimizers:
        opt_plot.data = exps_df.loc[
            (exps_df["opt"] == opt_plot.name)
            & (exps_df["momentum"] == opt_plot.momentum)
        ]

    eval_indices = [i for i in range(0, plotting_time + eval_every, eval_every)]
    timing_prefix = "steps" if using_step else "epoch"

    def make_filename(metric, plotting_time, logx, logy):
        suffix = ""
        match (logx, logy):
            case (False, False):
                suffix = "linear"
            case (False, True):
                suffix = "log"
            case (True, False):
                suffix = "logx"
            case (True, True):
                suffix = "loglog"

        return f"{metric}_best_{timing_prefix}_{plotting_time}_{suffix}.pdf"

    for measure in MEASURES:
        save_dir = base_save_dir / measure

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df_list = []
        metrics_used = []
        for _, metric in enumerate(metric_names):
            if metric not in exps_df.columns:
                continue
            make_plot_for_logy = lambda logx, logy: make_plot(
                optimizers=optimizers,
                eval_indices=eval_indices,
                experiments=experiments,
                group=group,
                main_metric=main_metric,
                plotting_time=plotting_time,
                using_step=using_step,
                measure=measure,
                metric=metric,
                save_path=save_dir
                / make_filename(metric, plotting_time, logy=logy, logx=logx),
                logx=logx,
                logy=logy,
                rel_width=rel_width,
                height_to_width_ratio=height_to_width_ratio,
                postprocess=postprocess,
            )

            val_dict = make_plot_for_logy(logx=False, logy=False)
            df_list.append(val_dict)

            if should_plot_logy(metric):
                make_plot_for_logy(logx=False, logy=True)
                make_plot_for_logy(logx=True, logy=True)
            metrics_used.append(metric)

        best_df = pd.DataFrame(df_list, index=metrics_used)
        best_df.to_csv(f"{save_dir}/{measure}.csv")


def make_plot(
    optimizers,
    eval_indices,
    experiments,
    group,
    main_metric,
    plotting_time,
    using_step,
    measure,
    metric,
    save_path,
    logx=False,
    logy=False,
    rel_width=0.5,
    height_to_width_ratio=1.0,
    postprocess=None,
):
    fig, ax = make_fig_axs(
        plt,
        rel_width=rel_width,
        height_to_width_ratio=height_to_width_ratio,
    )
    max_value = float("-inf")
    val_dict = {}
    for opt_plot in optimizers:
        if not opt_plot.data.empty:
            best_exps, _ = get_best(
                experiments, main_metric, opt_plot.data, measure=measure
            )

            df = _get_best_data(
                best_exps,
                metric,
                measure=measure,
                plotting_time=plotting_time,
                eval_indices=eval_indices,
                using_step=using_step,
            )

            val_dict[opt_plot.display_name] = df["middle"].iloc[-1]
            val = _plot_best_run_opt(df, ax, opt_plot, eval_indices=eval_indices)
            max_value = max(max_value, val)

    ylims = ax.get_ylim()
    if logy:
        ax.set_yscale("log")
        ax.set_ylim([ylims[0], 1.1 * ylims[1]])
    else:
        new_ylims = get_ylims_for_problem_linear(experiments[0], metric)
        if new_ylims is not None:
            ax.set_ylim(new_ylims)
        else:
            ax.set_ylim([0, 1.1 * ylims[1]])

    if logx:
        ax.set_xscale("log")

        def xdata_starts_at_0(line):
            x, y = line.get_data()
            return 0 in x

        should_shift_by_one = any([xdata_starts_at_0(line) for line in ax.get_lines()])
        if should_shift_by_one:
            for line in ax.get_lines():
                x, y = line.get_data()
                line.set_data([i + 1 for i in x], y)

    ax.set_xlabel(f"{'Steps' if using_step else 'Epochs'}")
    ax.set_ylabel(displayname(metric))
    ax.set_title(displayname(group))
    fig.tight_layout(pad=0)

    if postprocess is not None:
        postprocess(fig)

    print(f"Saving {save_path}")
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)

    return val_dict


def _plot_best_run_opt(
    best_exps_data: pd.DataFrame,
    ax: plt.axes,
    opt_plot: PlotOptimizer,
    eval_indices: List[int],
):
    opt_name = opt_plot.display_name
    line_color = opt_plot.line_color
    fill_color = opt_plot.fill_color
    linestyle = opt_plot.line_style

    import pdb

    #    pdb.set_trace()
    ax.plot(
        subsample(eval_indices),
        subsample(best_exps_data.loc[:, "middle"].values),
        label=opt_name,
        color=line_color,
        linestyle=linestyle,
    )

    ax.fill_between(
        subsample(eval_indices),
        subsample(best_exps_data.loc[:, "low"].values),
        subsample(best_exps_data.loc[:, "high"].values),
        color=fill_color,
        alpha=opt_plot.fill_alpha,
        linewidth=0.0,
    )

    return best_exps_data.max().max()


def _get_best_data(
    best_exps: List[Experiment],
    metric: str,
    measure: str,
    plotting_time: int,
    eval_indices: List[int],
    using_step: bool,
):
    values = []
    eval_every = eval_indices[1] - eval_indices[0]
    for exp in best_exps:
        data = exp.load_data()
        data = data.set_index("step" if using_step else "epoch")
        x = data.loc[eval_indices, metric]
        values.append(x)

    df = (
        pd.DataFrame(values)
        .transpose()
        .rename(columns={"0": "exp_0", "1": "exp_1", "2": "exp_2"})
    )

    valid_measures = ["min", "max", "median", "mean"]
    if measure not in valid_measures:
        raise ValueError(f"Measure {measure} invalid. Expected one of {valid_measures}")

    low, mid, high = df.min(axis=1), getattr(df, measure)(axis=1), df.max(axis=1)
    df = pd.concat([low, mid, high], axis=1)
    df = df.rename(columns={0: "low", 1: "middle", 2: "high"})

    return df
