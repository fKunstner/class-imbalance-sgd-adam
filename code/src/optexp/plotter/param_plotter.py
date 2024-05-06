import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from optexp import Experiment, config
from optexp.plotter.data_utils import (
    MEASURES,
    get_best,
    get_exps_data_epoch,
    get_measure_func,
    load_data_for_exps,
)
from optexp.plotter.names_and_consts import (
    metrics_to_plot_and_main_metric_for_standard_plots,
)
from optexp.plotter.style_lines import PlotOptimizer, get_optimizers
from optexp.utils import remove_duplicate_exps


@dataclass
class DataPoint:
    param_value: float
    experiments: List[Experiment]


@dataclass
class PlottingData:
    param: str
    points: List[DataPoint]
    save_folder_name: str


def plotter_cli(plotting_data: PlottingData):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        help="epoch used for plotting",
        default=None,
    )
    args = parser.parse_args()

    plot_param(plotting_data, epoch=args.epoch)


def plot_param(plotting_data: PlottingData, epoch: Optional[int] = None):
    problem = plotting_data.points[0].experiments[0].problem
    metrics, main_metric = metrics_to_plot_and_main_metric_for_standard_plots(problem)

    save_path_folder = (
        config.get_final_plots_directory() / plotting_data.save_folder_name
    )

    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)

    epoch = epoch if epoch is not None else find_max_viable_epoch(plotting_data)

    for data_point in plotting_data.points:
        exps = remove_duplicate_exps(data_point.experiments)
        data_point.experiments = exps

    exps_df = make_df_across_exps(plotting_data, metrics, epoch)

    for measure in MEASURES:
        save_path = save_path_folder / measure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot(
            plotting_data,
            exps_df,
            main_metric,
            metrics,
            measure=measure,
            save_path=save_path,
            epoch=epoch,
        )


def plot(
    plotting_data: PlottingData,
    exps_df: pd.DataFrame,
    main_metric: str,
    other_metrics: List[str],
    measure: str,
    save_path: Path,
    epoch: int,
):
    list_of_dict = []
    optimizers = get_optimizers()
    for data_point in plotting_data.points:
        exps = data_point.experiments
        val = data_point.param_value
        group = exps[0].group

        if not all(exp.group == group for exp in exps):
            raise ValueError("All experiments must have the same group.")

        for plot_opt in optimizers:
            df_opt = exps_df.loc[
                (exps_df["opt"] == plot_opt.name)
                & (exps_df["momentum"] == plot_opt.momentum)
                & (exps_df["group"] == group)
            ]

            if df_opt.empty:
                continue

            opt_dict = make_points(
                exps,
                plot_opt.name,
                main_metric,
                other_metrics,
                df_opt,
                momentum=plot_opt.momentum,
                measure=measure,
                epoch=epoch,
            )

            opt_dict.update({plotting_data.param: val})
            list_of_dict.append(opt_dict)

    plot_df = pd.DataFrame(list_of_dict)
    for plot_opt in optimizers:
        df = plot_df.loc[
            (plot_df["opt"] == plot_opt.name)
            & (plot_df["momentum"] == plot_opt.momentum)
        ]
        plot_opt.data = df

    for _, metric in enumerate(other_metrics):
        fig, ax = plt.subplots()
        max_value = float("-inf")
        for plot_opt in optimizers:
            if not plot_opt.data.empty:
                val = plot_param_opt(
                    plot_opt=plot_opt,
                    ax=ax,
                    param_name=plotting_data.param,
                    measure=measure,
                    metric=metric,
                )
                max_value = max(max_value, val)
        top_lim = 1.1 * max_value  # if "loss" not in metric.lower() else 2.0
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=top_lim)
        ax.set_xlabel(plotting_data.param)
        ax.set_ylabel(metric)
        ax.legend()
        ax.set_title(f"Optimizer Performance vs {plotting_data.param} at Epoch {epoch}")

        plot_save_path = save_path / f"{metric}_{epoch}.pdf"
        fig.tight_layout(pad=0)
        plt.savefig(plot_save_path)
        plt.close(fig)


def plot_param_opt(
    plot_opt: PlotOptimizer,
    ax: plt.axes,
    param_name: str,
    measure: str,
    metric: str,
) -> float:
    df = plot_opt.data
    if df is None:
        raise ValueError("Dataframe expected, got None")
    df_temp = get_points_to_plot(df, measure, metric)
    df = pd.concat([df_temp, df[param_name], df["lr"]], axis=1)
    linestyle = "solid" if plot_opt.momentum else "dashed"
    ax.plot(
        df[param_name].values,
        df.loc[:, "middle"].values,
        label=plot_opt.display_name,
        color=plot_opt.line_color,
        marker=".",
        linestyle=linestyle,
    )
    # for val in df["lr"].index:
    #     ax.annotate(
    #         f"{df.loc[val, 'lr']:.3f}",
    #         (df.loc[val, param_name], df.loc[val, "middle"]),
    #     )

    ax.fill_between(
        df[param_name].values,
        df.loc[:, "low"],
        df.loc[:, "high"],
        color=plot_opt.fill_color,
        linewidth=0.0,
    )
    return df[["low", "middle", "high"]].max().max()


def get_points_to_plot(df: pd.DataFrame, measure: str, metric: str):
    func = get_measure_func(measure)
    df = df[[f"exp_0_{metric}", f"exp_1_{metric}", f"exp_2_{metric}"]].apply(
        func,
        axis=1,
    )
    return df


def make_points(
    exps: List[Experiment],
    opt_name: str,
    main_metric: str,
    other_metrics: List[str],
    df: pd.DataFrame,
    momentum: bool,
    measure: str,
    epoch: int,
):
    best_exps, best_lr = get_best(exps, main_metric, df, measure)

    best_dict = {}
    for i, exp in enumerate(best_exps):
        try:
            exp_data = exp._load_wandb_data()
        except ValueError:
            exp_data = exp._load_local_data()
        all_metrics = [main_metric] + other_metrics
        for metric in all_metrics:
            first_valid_ind = exp_data.apply(pd.Series.first_valid_index)[metric]
            last_valid_ind = exp_data.apply(pd.Series.last_valid_index)[metric]
            try:
                metric_val = exp_data.loc[epoch, metric]
            except KeyError:
                metric_val = exp_data.loc[last_valid_ind, metric]
            initial_val = exp_data.loc[first_valid_ind, metric]
            if "loss" in metric.lower():
                if metric_val > 1.5 * initial_val:
                    metric_val = initial_val
            best_dict[f"exp_{i}_{metric}"] = metric_val
    best_dict["opt"] = opt_name
    best_dict["momentum"] = momentum
    best_dict["lr"] = best_lr
    return best_dict


def make_df_across_exps(
    plotting_data: PlottingData, metric_names: List[str], epoch: int
):
    dfs = []
    for data_point in plotting_data.points:
        exps_w_data = load_data_for_exps(data_point.experiments)
        exp_df = get_exps_data_epoch(exps_w_data, metric_names, epoch)
        dfs.append(exp_df)
    return pd.concat(dfs)


def find_max_viable_epoch(plotting_data: PlottingData):
    min_epoch = math.inf
    for data_point in plotting_data.points:
        for exp in data_point.experiments:
            min_epoch = min(min_epoch, exp.epochs)
    return min_epoch
