import dataclasses
import os
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from optexp import Experiment, MixedBatchSizeTextDataset, config
from optexp.plotter.data_utils import column_to_numpy, load_data_for_exps
from optexp.plotter.names_and_consts import (
    displayname,
    get_ylims_for_problem_linear,
    get_ylims_for_problem_log,
)
from optexp.plotter.plot_utils import normalize_y_axis, subsample
from optexp.plotter.style_figure import make_fig_axs
from optexp.plotter.style_lines import get_opt_plotting_config_for, get_optimizers
from optexp.utils import split_classes_by_frequency


def plot_per_class(
    experiments: List[Experiment],
    plotting_time: Optional[int] = None,
    using_step: Optional[bool] = False,
    where: Optional[Path] = None,
    rel_width: float = 1.0,
    height_to_width_ratio=0.8,
    postprocess=None,
    plot_overall_loss=True,
):
    data = load_data_for_exps(experiments)

    group = experiments[0].group

    if where is None:
        base_save_path = config.get_plots_directory() / Path(group) / Path("per-class")
    else:
        base_save_path = where

    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    dataset = experiments[0].problem.dataset

    if (
        "Dummy" in group
        or "LogReg" in group
        or "LinReg" in group
        or "ImbalancedMNIST_FB_Base_10k" in group
    ):
        dataset2 = dataclasses.replace(dataset, batch_size=500_000)
        out = dataset2.load()
        tr_load = out[0]
        va_load = out[1]
        X, y = next(iter(tr_load))
        X_va, y_va = next(iter(va_load))
    else:
        out = dataset.load()
        tr_load = out[0]
        va_load = out[1]
        ys = []
        for _, batch_y in tr_load:
            ys.append(batch_y)
        y = torch.cat(ys)
        ys_va = []
        for _, batch_y in va_load:
            ys_va.append(batch_y)
        y_va = torch.cat(ys_va)

    exp_dicts = [_["exp"] for _ in data]
    exp_data = [_["data"] for _ in data]
    targets = y
    targets_va = y_va

    possible_keys_perclass = [
        f"{tr_va}_{metric}"
        for tr_va in ["tr", "va", "val"]
        for metric in [
            "CrossEntropyLossPerClass",
            "MSELossPerClass",
            "AccuracyPerClass",
        ]
    ]
    possible_keys = [
        f"{tr_va}_{metric}"
        for tr_va in ["tr", "va", "val"]
        for metric in [
            "CrossEntropyLoss",
            "ClassificationSquaredLoss",
            "Accuracy",
        ]
    ]

    for i in range(len(exp_data)):
        for key in possible_keys_perclass:
            if key not in exp_data[i].columns:
                continue
            exp_data[i][key] = exp_data[i][key].apply(column_to_numpy)

    targets_np = targets.cpu().numpy()
    targets_np_va = targets_va.cpu().numpy()
    bincounts = np.bincount(targets_np)
    bincounts_va = np.bincount(targets_np_va)

    groups = None
    groups_va = None
    if "ImbalancedMNIST" in group:
        mnist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        other = list(set(np.unique(targets_np)) - set(mnist))
        groups = [(np.array(mnist, dtype=np.int64), np.array(other, dtype=np.int64))]
    elif "TEnc" in group:
        groups = [
            split_classes_by_frequency(targets_np, n_splits=i) for i in [2, 5, 10]
        ]
        groups_va = [
            split_classes_by_frequency(targets_np_va, n_splits=i) for i in [2, 5, 10]
        ]
    else:
        groups = [
            split_classes_by_frequency(targets_np, n_splits=i)
            for i in [2, 3, 4, 5, 10, 11]
        ]
        groups_va = [
            split_classes_by_frequency(targets_np_va, n_splits=i)
            for i in [2, 3, 4, 5, 10, 11]
        ]

    for key, key_perclass in zip(possible_keys, possible_keys_perclass):
        if key not in exp_data[0].columns and key_perclass not in exp_data[0].columns:
            continue

        for log_yscale in [True, False]:

            groups_to_use = groups if "tr_" in key else groups_va

            for idx_for_groups in groups_to_use:
                if len(idx_for_groups[0]) == 0:
                    # The data doesn't have enough classes to split in that many groups
                    continue
                if group == "LogReg_Synthetic_Per_Class" and "va" in key:
                    # That problem doesn't have a validation set
                    continue

                bincounts_to_use = bincounts if "tr_" in key else bincounts_va

                filename = f"{key}_{'log' if log_yscale else 'linear'}_{len(idx_for_groups)}.pdf"
                make_figure(
                    base_save_path / filename,
                    exp_data,
                    exp_dicts,
                    idx_for_groups,
                    bincounts_to_use,
                    key,
                    key_perclass,
                    log_yscale=log_yscale,
                    max_plotting_time=plotting_time,
                    using_step=using_step,
                    rel_width=rel_width,
                    height_to_width_ratio=height_to_width_ratio,
                    postprocess=postprocess,
                    plot_overall_loss=plot_overall_loss,
                )


def make_figure(
    base_save_path,
    exp_data,
    exp_dicts,
    idx_for_groups,
    bincounts,
    key,
    key_perclass,
    log_yscale=False,
    max_plotting_time=None,
    using_step=False,
    rel_width=1.0,
    height_to_width_ratio=1.0,
    plot_overall_loss=True,
    postprocess=None,
):
    split_mom_nomom = len(exp_dicts) > 4 and len(exp_dicts) % 2 == 0

    overall_loss_offset = 1 if plot_overall_loss else 0
    if split_mom_nomom:
        fig, axes = make_fig_axs(
            plt,
            nrows=2,
            ncols=int(len(exp_data) / 2) + overall_loss_offset,
            height_to_width_ratio=height_to_width_ratio,
            rel_width=rel_width,
        )
        idx_rows = [
            [
                i
                for i, exp in enumerate(exp_dicts)
                if not get_opt_plotting_config_for(exp).momentum
            ],
            [
                i
                for i, exp in enumerate(exp_dicts)
                if get_opt_plotting_config_for(exp).momentum
            ],
        ]
    else:
        fig, axes = make_fig_axs(
            plt,
            nrows=1,
            ncols=len(exp_data) + overall_loss_offset,
            height_to_width_ratio=height_to_width_ratio,
            rel_width=rel_width,
        )
        if isinstance(axes, Axes):
            axes = [[axes]]
        idx_rows = [list(range(len(exp_dicts))), []]

    if key in exp_data[0].columns:
        for row in range(len(axes)):
            for i, (exp_def, exp_df) in enumerate(zip(exp_dicts, exp_data)):
                if i not in idx_rows[row]:
                    continue

                plot_config = get_opt_plotting_config_for(exp_def)

                if using_step:
                    time = (
                        max_plotting_time
                        if max_plotting_time is not None
                        else exp_df["step"].max()
                    )
                    exp_df = exp_df.set_index("step", drop=False)
                    time_method = "step"
                else:
                    time = (
                        max_plotting_time
                        if max_plotting_time is not None
                        else len(exp_df["epoch"])
                    )
                    exp_df = exp_df.set_index("epoch", drop=False)
                    time_method = "epoch"

                #                import pdb
                #                pdb.set_trace()
                if plot_overall_loss:
                    axes[row][0].plot(
                        subsample(
                            exp_df.loc[:time, time_method], NMAX=1000, linear_only=True
                        ),
                        subsample(exp_df.loc[:time, key], NMAX=1000, linear_only=True),
                        #                    subsample(exp_df["epoch"][:epoch]),
                        #                    subsample(exp_df[key][:epoch]),
                        label=plot_config.display_name,
                        color=plot_config.line_color,
                        linestyle=plot_config.line_style,
                    )
    if key_perclass in exp_data[0].columns:
        for row in range(len(axes)):
            for i, (exp_def, exp_df) in enumerate(zip(exp_dicts, exp_data)):
                if i not in idx_rows[row]:
                    continue
                plot_on(
                    axes[row][overall_loss_offset + idx_rows[row].index(i)],
                    exp_def,
                    exp_df,
                    idx_for_groups,
                    bincounts,
                    key_perclass,
                    max_time=max_plotting_time,
                    using_step=using_step,
                )

    for row in range(len(axes)):
        axes[row][0].set_ylabel(displayname(key))
    if plot_overall_loss:
        axes[0][0].set_title("Overall loss")

    axes_flat = list([ax for row in axes for ax in row])

    for ax in axes[-1]:
        ax.set_xlabel(f"{'Step' if using_step else 'Epoch'}")

    if log_yscale:
        for ax in axes_flat:
            ax.set_yscale("log")
        ylims = get_ylims_for_problem_log(exp_dicts[0], key)
        if ylims is not None:
            for ax in axes_flat:
                ax.set_ylim(ylims)
    else:
        ylims = get_ylims_for_problem_linear(exp_dicts[0], key)
        if ylims is not None:
            for ax in axes_flat:
                ax.set_ylim(ylims)

    normalize_y_axis(*axes_flat)
    fig.tight_layout(pad=0.1)
    if postprocess is not None:
        postprocess(fig)
    print("Saving", base_save_path)
    plt.savefig(base_save_path)
    plt.close(fig)


def plot_on(
    ax, exp_def, exp_df, idx_for_groups, bincounts, thing_to_plot, max_time, using_step
):
    correction_factor = 1.0
    if "Dummy_LinReg" in exp_def.group and "MSELoss" in thing_to_plot:
        correction_factor = 4000

    as_array = np.stack(exp_df[thing_to_plot].values) * correction_factor
    if len(bincounts) < as_array.shape[1]:
        bincounts = np.concatenate(
            [bincounts, np.zeros(as_array.shape[1] - bincounts.shape[0])], axis=0
        )
    as_array = as_array * bincounts

    cmap = matplotlib.cm.get_cmap("viridis")
    plotting_config = [
        {"color": cmap(interp)} for interp in np.linspace(0, 1, num=len(idx_for_groups))
    ]

    #    import pdb
    #    pdb.set_trace()

    if not any(0 in group for group in idx_for_groups) and not np.all(
        np.isnan(as_array[:, 0])
    ):

        raise ValueError("Something went wrong. Index 0 is not in any group.")

    for i, idx_for_group in enumerate(idx_for_groups):
        #        epoch = max_epoch if max_epoch is not None else len(exp_df["epoch"])

        if using_step:
            time = max_time if max_time is not None else exp_df["step"].max()
            exp_df = exp_df.set_index("step", drop=False)
            time_method = "step"
        else:
            time = max_time if max_time is not None else len(exp_df["epoch"])
            exp_df = exp_df.set_index("epoch", drop=False)
            time_method = "epoch"

        xs = exp_df.loc[:time, time_method]

        reduced_array = as_array[: len(xs)]
        values = np.nansum(reduced_array[:, idx_for_group], axis=1) / np.sum(
            bincounts[idx_for_group]
        )

        ax.plot(
            subsample(xs),
            subsample(values),
            **plotting_config[i],
            label=f"{i+1}",
        )
    ax.set_title(f"{get_opt_plotting_config_for(exp_def).name}")


def shift_labels_idx(exp_df):
    pass
