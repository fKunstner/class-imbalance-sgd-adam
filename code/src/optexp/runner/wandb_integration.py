from typing import List

import numpy as np
import pandas as pd
import wandb

from optexp import config
from optexp.config import get_logger, get_wandb_timeout
from optexp.plotter.data_utils import column_to_numpy, should_convert_column_to_numpy


class WandbAPI:
    """Static class to provide a singleton handler to the wandb api.

    When in need to call the Wandb API, use WandbAPI.get_handler()
    instead of creating a new instance of wandb.Api().
    """

    api_handler = None

    @staticmethod
    def get_handler():
        if WandbAPI.api_handler is None:
            WandbAPI.api_handler = wandb.Api(timeout=get_wandb_timeout())
        return WandbAPI.api_handler

    @staticmethod
    def get_path():
        return f"{config.get_wandb_entity()}/{config.get_wandb_project()}"


def get_wandb_runs_for_group(group: str) -> List[wandb.apis.public.Run]:
    """Get the runs of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = WandbAPI.get_handler().runs(
        WandbAPI.get_path(), filters={"group": group}, per_page=1000
    )

    if any("exp_id" not in run.config for run in runs):
        get_logger().warning("Some runs do not have an exp_id attribute.")

    return [run for run in runs if run.state == "finished"]


def get_successful_ids_and_runs(group: str):
    """Get the experiment ids of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = get_wandb_runs_for_group(group)
    successful_runs = []
    successful_exp_ids = []
    for run in runs:
        if run.config["exp_id"] not in successful_exp_ids and run.state == "finished":
            successful_runs.append(run)
            successful_exp_ids.append(run.config["exp_id"])

    return successful_exp_ids, successful_runs


def numpyify(df: pd.DataFrame):
    df.replace("Infinity", np.inf, inplace=True)
    for key in df.columns:
        if should_convert_column_to_numpy(df[key]):
            df[key] = df[key].apply(column_to_numpy)
    return df


def download_run_data(run: wandb.apis.public.Run, parquet=True):
    """Given a Wandb Run, download the full history."""
    save_dir = config.get_wandb_cache_directory() / run.config["exp_config"]["group"]
    if parquet:
        save_file = save_dir / f"{run.config['exp_id']}.parquet"
    else:
        save_file = save_dir / f"{run.config['exp_id']}.csv"

    if save_file.exists():
        return

    df = run.history(pandas=True, samples=10000)
    if parquet:
        df = numpyify(df)
        df.to_parquet(save_file)
    else:
        df.to_csv(save_file)


def download_summary(group=None):
    """Download a summary of all runs on the wandb project."""
    save_file = config.get_wandb_cache_directory() / group / "summary.csv"

    if save_file.exists():
        return

    filters = {"group": group} if group is not None else {}
    runs = WandbAPI.get_handler().runs(
        config.get_wandb_project(), filters=filters, per_page=1000
    )

    configs = []
    systems = []
    miscs = []

    for run in runs:
        configs.append(flatten_dict(run.config))
        systems.append(flatten_dict(run._attrs["systemMetrics"]))
        miscs.append(
            {
                "name": run.name,
                "id": run.id,
                "group": run.group,
                "state": run.state,
                "tags": run.tags,
                "histLineCount": run._attrs["historyLineCount"],
            }
        )

    misc_df = pd.DataFrame.from_records(miscs)
    config_df = pd.DataFrame.from_records(configs)
    system_df = pd.DataFrame.from_records(systems)
    all_df = pd.concat([misc_df, config_df, system_df], axis=1)

    all_df.to_csv(save_file)


def flatten_dict(x):
    return pd.io.json._normalize.nested_to_record(x)
