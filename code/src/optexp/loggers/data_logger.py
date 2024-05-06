from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
import wandb

from optexp import config
from optexp.config import get_logger
from optexp.loggers.rate_limited_logger import RateLimitedLogger
from optexp.loggers.utils import pprint_dict


class DataLogger:
    def __init__(
        self,
        config_dict: Dict,
        group: str,
        run_id: str,
        exp_id: str,
        save_directory: Path,
        wandb_autosync: bool = True,
    ) -> None:
        """Data logger for experiments.

        Delegates to a console logger to print progress.
        Saves the data to a csv and experiment configuration to a json file.
        Creates the save_dir if it does not exist.

        Args:
            run_id: Unique id for the run
                (an experiment might have multiple runs)
            experiment: The experiment to log
        """
        self.run_id = run_id
        self.config_dict = config_dict
        self.save_directory = save_directory
        self.wandb_autosync = wandb_autosync
        print(self.config_dict)

        self._dicts: List[Dict] = []
        self._current_dict: Dict = {}
        self.console_logger = RateLimitedLogger()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        if config.get_wandb_status():
            get_logger().info("WandB is enabled")
            if config.get_wandb_key() is not None:
                self.run = wandb.init(
                    project=config.get_wandb_project(),
                    entity=config.get_wandb_entity(),
                    config={
                        "exp_id": exp_id,
                        "run_id": run_id,
                        "exp_config": self.config_dict,
                    },
                    group=group,
                    mode=config.get_wandb_mode(),
                    dir=config.get_experiment_directory(),
                )
            else:
                raise ValueError("WandB API key not set.")
            if self.run is None:
                raise ValueError("WandB run initialization failed.")
            get_logger().info(f"--- WANDB initialized. Run ID: {self.run.id}")
        else:
            get_logger().info("WandB is NOT enabled.")

        self.handler = config.set_logfile(save_directory / f"{run_id}.log")

    def log_data(self, metric_dict: dict) -> None:
        """Log a dictionary of metrics.

        Based on the wandb log function (https://docs.wandb.ai/ref/python/log)
        Uses the concept of "commit" to separate different steps/iterations.

        log_data can be called multiple times per step,
        and repeated calls update the current logging dictionary.
        If metric_dict has the same keys as a previous call to log_data,
        the keys will get overwritten.

        To move on to the next step/iteration, call commit.

        Args:
            metric_dict: Dictionary of metrics to log
        """
        self._current_dict.update(metric_dict)
        if config.get_wandb_status():
            wandb.log(metric_dict, commit=False)

    def commit(self) -> None:
        """Commit the current logs and move on to the next step/iteration."""
        self.console_logger.log(pprint_dict(self._current_dict))
        self._dicts.append(copy.deepcopy(self._current_dict))
        self._current_dict = {}
        if config.get_wandb_status():
            wandb.log({}, commit=True)

    def save(self, exit_code) -> None:
        """Save the experiment configuration and results to disk."""
        filepath_csv = self.save_directory / f"{self.run_id}.csv"
        filepath_json = self.save_directory / f"{self.run_id}.json"

        get_logger().info(f"Saving experiment configs to {filepath_json}")

        json_data = json.dumps(self.config_dict, indent=4)
        with open(filepath_json, "w") as outfile:
            outfile.write(json_data)

        get_logger().info(f"Saving experiment results to {filepath_csv}")

        data_df = pd.DataFrame.from_records(self._dicts)
        data_df.to_csv(filepath_csv)
        try:
            get_logger().info(f"Last saved dict: {pprint_dict(self._dicts[-1])}")
        except IndexError:
            get_logger().debug(f"No info to log.")

        if config.get_wandb_status():
            if self.run is None:
                raise ValueError("Expected a WandB run but None found.")

            get_logger().info("Finishing Wandb run")
            wandb.finish(exit_code=exit_code)

            if config.get_wandb_mode() == "offline" and self.wandb_autosync:
                get_logger().info(f"Uploading wandb run in {Path(self.run.dir).parent}")
                command = (
                    f"wandb sync "
                    f"--id {self.run.id} "
                    f"-p {config.get_wandb_project()} "
                    f"-e {config.get_wandb_entity()} "
                    f"{Path(self.run.dir).parent}"
                )
                get_logger().info(f"    {command}")

                subprocess.run(
                    command,
                    shell=True,
                )
            else:
                command = (
                    f"wandb sync "
                    f"--id {self.run.id} "
                    f"-p {config.get_wandb_project()} "
                    f"-e {config.get_wandb_entity()} "
                    f"{Path(self.run.dir).parent}"
                )
                get_logger().info(f"Not Running:    {command}")

        config.remove_loghandler(handler=self.handler)
