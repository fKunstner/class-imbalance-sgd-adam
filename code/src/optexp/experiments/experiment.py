import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from optexp import config
from optexp.config import get_logger
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.optimizers import Optimizer
from optexp.problems import DivergingException, Problem


@dataclass
class Experiment:
    """
    Represents an experiment where a problem is optimized given an optimizer.

    Attributes:
        optim: The optimizer to use for optimizing the model defined in the problem.
        problem: The problem to optimize.
        group: The group of experiments this experiment belongs to.
        seed: The seed to use.
        epochs: The number of epochs the problem is optimized.
    """

    optim: Optimizer
    problem: Problem
    group: str
    seed: int
    epochs: int

    @staticmethod
    def generate_experiments_from_opts_and_seeds(
        opts_and_seeds: List[Tuple[List[Optimizer], List[int]]],
        problem: Problem,
        epochs: int,
        group: str,
    ):
        return sum(
            [
                Experiment.generate_all(opts, [problem], seeds, [epochs], group)
                for (opts, seeds) in opts_and_seeds
            ],
            [],
        )

    @staticmethod
    def generate_all(
        opts: List[Optimizer],
        probs: List[Problem],
        seeds: List[int],
        epochs: List[int],
        group: str,
    ):
        return [
            Experiment(optim=opt, problem=prob, group=group, seed=seed, epochs=epoch)
            for opt in opts
            for prob in probs
            for seed in seeds
            for epoch in epochs
        ]

    def run_experiment(self) -> None:
        """
        Performs a run of the experiment. Generates the run-id, applies the seed
        and creates the data logger. Initializes the problem and optimizer and
        optimizes the problem given the optimizer for the defined amount of epochs.
        Logs the loss function values/metrics returned during the eval and training.
        Catches any exception raised during this process and logs it before exiting.

        Raises:
            BaseException: Raised when user Ctrl+C when experiment is running.
        """
        run_id = time.strftime("%Y-%m-%d--%H-%M-%S")

        self._apply_seed()

        data_logger = DataLogger(
            config_dict=asdict_with_class(self),
            group=self.group,
            run_id=run_id,
            exp_id=self.exp_id(),
            save_directory=self.save_directory(),
        )

        get_logger().info("=" * 80)
        get_logger().info(f"Initializing  experiment: {self}")
        get_logger().info("=" * 80)

        try:
            self.problem.init_problem()
            opt = self.optim.load(self.problem.torch_model)

            metrics_eval_val = self.problem.eval(val=True)
            metrics_eval_train = self.problem.eval(val=False)

            data_logger.log_data(metrics_eval_val)
            data_logger.log_data(metrics_eval_train)
            data_logger.log_data({"epoch": 0})
            data_logger.commit()

            for e in range(1, self.epochs + 1):
                metrics_training = self.problem.one_epoch(opt)
                metrics_eval_train = self.problem.eval(val=False)
                metrics_eval_val = self.problem.eval(val=True)

                data_logger.log_data({"epoch": e})
                data_logger.log_data(metrics_training)
                data_logger.log_data(metrics_eval_train)
                data_logger.log_data(metrics_eval_val)

                data_logger.commit()
        except DivergingException as e:
            get_logger().warning("TERMINATING EARLY. Diverging.")
            get_logger().warning(e, exc_info=True)
            data_logger.save(exit_code=0)
            return
        except Exception as e:
            get_logger().error("TERMINATING. Encountered error")
            get_logger().error(e, exc_info=True)
            data_logger.save(exit_code=1)
            return
        except BaseException as e:
            get_logger().error("TERMINATING. System exit")
            get_logger().error(e, exc_info=True)
            data_logger.save(exit_code=1)
            raise e
        get_logger().info("Experiment finished.")
        data_logger.save(exit_code=0)

    def _apply_seed(self) -> None:
        """Apply the seed to all random number generators.

        To be called before the experiment is run.
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def save_directory(self) -> Path:
        """Return the directory where the experiment results are saved."""
        base = config.get_experiment_directory()
        exp_dir = (
            f"{self.problem.__class__.__name__}_"
            f"{self.problem.model.__class__.__name__}_"
            f"{self.problem.dataset.name}"
        )
        save_dir = base / exp_dir / self.exp_id()
        return save_dir

    def exp_id(self) -> str:
        """Return a unique identifier for this experiment.

        Not a unique identifier for the current run of the experiment.
        Should be unique for the definition of the experiment, combining
        the problem, optimizer, and seed.
        """
        return hashlib.sha1(str.encode(str(self))).hexdigest()

    def load_data(self):
        """Tries to load any data for the experiment.

        Starts by trying to load data from the wandb download folder,
        if that fails it tries to load data from the local runs folder.
        """
        try:
            df = self._load_wandb_data()
        except ValueError as _:
            print(f"Experiment did not have wandb data for, trying local data [{self}]")
            df = self._load_local_data()
        return df

    def _load_local_data(self) -> Optional[pd.DataFrame]:
        """Loads the most recent experiment run data saved locally."""
        save_dir = self.save_directory()
        # get the timestamps of the runs from the names of the files
        time_stamps = [
            time.strptime(str(Path(x).stem), "%Y-%m-%d--%H-%M-%S")
            for x in os.listdir(save_dir)
        ]

        if time_stamps is None:
            return None

        most_recent_run = max(time_stamps)
        csv_file_path = (
            save_dir / f"{time.strftime('%Y-%m-%d--%H-%M-%S', most_recent_run)}.csv"
        )
        run_data = pd.read_csv(csv_file_path)
        return run_data

    def _load_wandb_data(self) -> pd.DataFrame:
        """Loads data from most recent run of experiment from wandb"""
        save_dir = (
            config.get_wandb_cache_directory()
            / Path(self.group)
            / f"{self.exp_id()}.parquet"
        )
        if not save_dir.is_file():
            raise ValueError(
                f"Experiment data has not been downloaded for exp:{str(self)}"
            )

        run_data = pd.read_parquet(save_dir)
        return run_data
