import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from optexp import Experiment, LightningExperiment, Problem, config
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.plotter.data_utils import get_exps_data_epoch, load_data_for_exps
from optexp.plotter.names_and_consts import (
    displayname,
    metrics_to_plot_and_main_metric_for_standard_plots,
)
from optexp.plotter.style_figure import make_fig_axs
from optexp.plotter.style_lines import PlotOptimizer, get_optimizers


def plot_grids(
    experiments: List[Experiment],
    plotting_time: Optional[int] = None,
    using_step: Optional[bool] = False,
) -> None:
    group = experiments[0].group
    problem = experiments[0].problem

    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")
    if not all(exp.problem == problem for exp in experiments) and not isinstance(experiments[0], LightningExperiment):
        raise ValueError("All experiments must have the same problem.")

    if using_step:
        if isinstance(experiments[0], LightningExperiment):
            plotting_time = plotting_time if plotting_time else experiments[0].steps
        else:
            raise ValueError("Only LightningExperiment supports plotting by step.")
    else:
        if isinstance(experiments[0], LightningExperiment):
            raise ValueError("LightningExperiment don't support epoch experiments")
        else:
            plotting_time = plotting_time if plotting_time else experiments[0].epochs

    json_data = json.dumps(asdict_with_class(problem), indent=4)
    save_path = config.get_plots_directory() / Path(group) / Path("grids")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    optimizers = get_optimizers()
    _, metric_names = metrics_to_plot_and_main_metric_for_standard_plots(problem)

    exps_w_data = load_data_for_exps(experiments)
    exps_df = get_exps_data_epoch(exps_w_data, metric_names, plotting_time, using_step)

    for opt_plot in optimizers:
        opt_data = exps_df.loc[
            (exps_df["opt"] == opt_plot.name)
            & (exps_df["momentum"] == opt_plot.momentum)
        ]
        opt_plot.data = opt_data

    for _, metric in enumerate(metric_names):
        if metric not in exps_df.columns:
            continue
        
        fig, ax = make_fig_axs(plt, rel_width=0.5)
        max_value = float("-inf")
        for opt_plot in optimizers:
            if not opt_plot.data.empty:
                val = plot_opt_grid(
                    opt_plot.data,
                    ax,
                    metric,
                    opt_plot,
                )
                max_value = max(max_value, val)

        if "loss" in metric.lower():
            ax.set_yscale("log")
            ylims = ax.get_ylim()
            ax.set_ylim([ylims[0], max_value * 1.1])
        else:
            ax.set_ylim([0, max_value * 1.1])

        ax.set_xlabel("Learning Rates")
        ax.set_ylabel(displayname(metric))

        y_ax_label = "Step" if using_step else "Epoch"
        ax.set_title(f"Gridsearch at {y_ax_label} {plotting_time}")

        fig.tight_layout(pad=0)
        filepath = save_path / f"{metric}_{plotting_time}.pdf"
        plt.savefig(filepath)
        print(f"Saving {filepath}")
        plt.clf()
        plt.close(fig)

    with open(save_path.parent / "problem.json", "w") as outfile:
        outfile.write(json_data)


def plot_opt_grid(
    opt_dataframe: pd.DataFrame,
    ax: plt.axes,
    metric_name: str,
    opt_plot: PlotOptimizer,
) -> float:
    opt_name = opt_plot.display_name
    line_color = opt_plot.line_color
    fill_color = opt_plot.fill_color
    linestyle = opt_plot.line_style

    df = opt_dataframe.groupby(["lr"]).apply(
        lambda x: pd.Series(
            [
                x[metric_name].min(),
                x[metric_name].mean(),
                x[metric_name].max(),
            ],
            index=["min", "mean", "max"],
        )
    )
    ax.set_xscale("log")
    lrs = df.index.values

    ax.plot(
        lrs,
        df.loc[:, "mean"].values,
        label=opt_name,
        color=line_color,
        marker=".",
        linestyle=linestyle,
    )
    ax.fill_between(
        lrs,
        df.loc[:, "min"],
        df.loc[:, "max"],
        color=fill_color,
        alpha=opt_plot.fill_alpha,
        linewidth=0.0,
    )
    return df.max().max()
