import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from optexp import Experiment

MEASURES = ["max"]  # ["max", "median"]  # ["min", "mean", "median", "max"]


def load_data_for_exps(experiments: List[Experiment]):
    exps_with_data = []
    for exp in experiments:
        df = exp.load_data()
        exp_w_data = {"exp": exp, "data": df}
        exps_with_data.append(exp_w_data)
    return exps_with_data


def get_exps_data_epoch(
    exps_w_data: List[Dict],
    metric_names: List[str],
    plotting_time: int,
    using_step: Optional[bool] = False,
) -> pd.DataFrame:
    list_of_dicts = []
    for exp_w_data in exps_w_data:
        exp = exp_w_data["exp"]
        if using_step and exp.steps < plotting_time:
            raise ValueError(
                f"The following experiment in this group {exp.group}"
                f"has only {exp.steps} epochs."
                f"But you want to plot at {plotting_time} epochs: {str(exp)}"
            )
        elif (not using_step) and exp.epochs < plotting_time:
            raise ValueError(
                f"The following experiment in this group {exp.group}"
                f"has only {exp.epochs} epochs."
                f"But you want to plot at {plotting_time} epochs: {str(exp)}"
            )

        metric_dict = {
            "seed": exp.seed,
            "lr": exp.optim.learning_rate.as_float(),
            "exp_id": exp.exp_id(),
            "group": exp.group,
            "opt": exp.optim.__class__.__name__,
        }
        if (
            exp.optim.__class__.__name__ == "Adam"
            or exp.optim.__class__.__name__ == "AdamW"
        ):
            metric_dict["momentum"] = True if exp.optim.beta1 != 0.0 else False
        elif exp.optim.__class__.__name__ == "Adagrad":
            metric_dict["momentum"] = False
        elif exp.optim.__class__.__name__ == "LineSearchSGD":
            metric_dict["momentum"] = False
        else:
            metric_dict["momentum"] = True if exp.optim.momentum != 0.0 else False
        df = exp_w_data["data"]

        first_valid_indices = df.apply(lambda series: series.first_valid_index())

        if len(df.index) == 0:
            warnings.warn(
                f"Experiment with ID {exp.exp_id()} has no data. Run diverged?"
            )
            continue

        # set index to either steps or epochs, current version just uses that epoch and index are  the same
        if using_step:
            df = df.set_index("step")
        else:
            df = df.set_index("epoch")

        for metric in metric_names:
            if metric not in df.columns:
                continue
            if plotting_time > df.index.max():
                # epoch we want to plot is after the experiment terminated (in case of early termination)
                # pick starting value
                metric_dict[metric] = df.loc[first_valid_indices[metric], metric]
            else:
                if math.isnan(df.loc[plotting_time, metric]) or math.isinf(
                    df.loc[plotting_time, metric]
                ):
                    # experiment terminated at exactly the epoch
                    metric_dict[metric] = df.loc[first_valid_indices[metric], metric]
                else:
                    if "accuracy" in metric.lower():
                        if (
                            df.loc[plotting_time, metric]
                            < df.loc[first_valid_indices[metric], metric] * 0.75
                        ):
                            # threshold low accuracy values
                            metric_dict[metric] = df.loc[
                                first_valid_indices[metric], metric
                            ]
                        else:
                            metric_dict[metric] = df.loc[plotting_time, metric]
                    else:
                        if (
                            df.loc[plotting_time, metric]
                            > df.loc[first_valid_indices[metric], metric] * 1.5
                        ):
                            # threshold high loss values
                            metric_dict[metric] = df.loc[
                                first_valid_indices[metric], metric
                            ]
                        else:
                            metric_dict[metric] = df.loc[plotting_time, metric]
        list_of_dicts.append(metric_dict)

    return pd.DataFrame(list_of_dicts)


def get_best(
    exps: List[Experiment],
    metric: str,
    opt_df: pd.DataFrame,
    measure: str,
):
    if measure == "mean":
        # returns best learning rate since index of dataframe is learning rates
        best_lr = opt_df.groupby(["lr"])[metric].mean().idxmin()
        best_exp_ids = opt_df.loc[opt_df["lr"] == best_lr]["exp_id"].values
    elif measure == "min":
        best_lr = opt_df.groupby(["lr"])[metric].min().idxmin()
        best_exp_ids = opt_df.loc[opt_df["lr"] == best_lr]["exp_id"].values
    elif measure == "max":
        best_lr = opt_df.groupby(["lr"])[metric].max().idxmin()
        best_exp_ids = opt_df.loc[opt_df["lr"] == best_lr]["exp_id"].values
    elif measure == "median":
        best_lr = opt_df.groupby(["lr"])[metric].median().idxmin()
        best_exp_ids = opt_df.loc[opt_df["lr"] == best_lr]["exp_id"].values
    else:
        raise ValueError(f"{measure} is valid: {MEASURES}")

    best_exps = [exp for exp in exps if exp.exp_id() in best_exp_ids]
    return best_exps, best_lr


def get_measure_func(measure: str):
    if measure == "min":
        return lambda x: pd.Series(
            [
                x.min(),
                x.min(),
                x.max(),
            ],
            index=["low", "middle", "high"],
        )
    elif measure == "max":
        return lambda x: pd.Series(
            [
                x.min(),
                x.max(),
                x.max(),
            ],
            index=["low", "middle", "high"],
        )
    elif measure == "median":
        return lambda x: pd.Series(
            [
                x.min(),
                x.median(),
                x.max(),
            ],
            index=["low", "middle", "high"],
        )
    elif measure == "mean":
        return lambda x: pd.Series(
            [
                x.min(),
                x.mean(),
                x.max(),
            ],
            index=["low", "middle", "high"],
        )
    else:
        raise ValueError(f"{measure} is valid: {MEASURES}")


def should_convert_column_to_numpy(series: pd.Series):
    def is_string_repr_of_array(entry):
        if isinstance(entry, str):
            if entry.startswith("[") and entry.endswith("]"):
                return True
        return False

    def is_list_of_elements_mostly_floats(entry):
        def list_elem_is_float_or_str_nan(list_element):
            return (
                isinstance(list_element, float)
                or isinstance(list_element, int)
                or (isinstance(list_element, str) and list_element == "NaN")
            )

        if isinstance(entry, list):
            if all([list_elem_is_float_or_str_nan(elem) for elem in entry]):
                return True
        return False

    if len(series) > 1:
        val = series[1]
    else:
        val = series[0]

    if is_string_repr_of_array(val) or is_list_of_elements_mostly_floats(val):
        return True
    else:
        return False


def column_to_numpy(x):
    """Convert string repr of numpy arrays to numpy arrays."""

    def convert_str_to_numpy(str_repr):
        str_repr = str_repr.strip("[]")
        str_repr = str_repr.replace("'NaN'", "nan")
        return np.fromstring(str_repr, sep=", ", dtype=float)

    def convert_list_to_numpy(list_repr):
        list_repr = [np.inf if x == "Infinity" else x for x in list_repr]
        return np.array(list_repr, dtype=np.float32)

    if x is None:
        return np.nan
    if type(x) is float or type(x) is int or type(x) is np.ndarray:
        return x
    elif type(x) is str:
        return convert_str_to_numpy(x)
    elif isinstance(x, list):
        return convert_list_to_numpy(x)
    else:
        raise ValueError(f"Cannot convert row, unknown type {type(x)}")
