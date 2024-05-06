import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree
from typing import List, Optional

from tqdm import tqdm

from optexp import config
from optexp.config import get_logger
from optexp.datasets import download_dataset
from optexp.experiments.experiment import Experiment
from optexp.experiments.lightning_experiment import LightningExperiment
from optexp.plotter.best_run_plotter import plot_best
from optexp.plotter.grid_plotter import plot_grids
from optexp.plotter.plot_per_class import plot_per_class
from optexp.runner.slurm.sbatch_writers import (
    make_jobarray_file_contents,
    make_jobarray_file_contents_split,
)
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.runner.wandb_integration import (
    download_run_data,
    download_summary,
    get_successful_ids_and_runs,
    get_wandb_runs_for_group,
)
from optexp.utils import remove_duplicate_exps


def exp_runner_cli(
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,
    python_file: Optional[Path] = None,
) -> None:
    """Command line interface for running experiments.

    Args:
        experiments: List of experiments to run
        slurm_config: Configuration to use for running experiments on Slurm
        python_file: Path to the python file to run the experiments from.
            Defaults to the file that called this function, sys.argv[0]
    """
    experiments = remove_duplicate_exps(experiments)
    parser = argparse.ArgumentParser()
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--report",
        metavar="KEY",
        nargs="?",
        default=False,
        const=True,
        type=str,
        help="Generates a report on what experiments have been run/are stored on wandb."
        + "If a key is provided, the report will be grouped by that key.",
    )
    action_group.add_argument(
        "--local",
        "--run-local",
        action="store_true",
        help="Run experiments locally.",
        default=False,
    )
    action_group.add_argument(
        "--slurm",
        "--run-slurm",
        action="store_true",
        help="Run experiments on Slurm.",
        default=False,
    )
    action_group.add_argument(
        "--slurm_split",
        type=int,
        action="store",
        help="Run all experiments on same machine on Slurm.",
        default=None,
    )
    action_group.add_argument(
        "--single",
        "--run-single-locally",
        action="store",
        type=int,
        help="Run a single experiment locally, by index.",
        default=None,
    )
    action_group.add_argument(
        "--split_index",
        action="store",
        type=int,
        help="Run all experiments locally.",
        default=None,
    )
    parser.add_argument(
        "--split_num",
        action="store",
        type=int,
        help="Run all experiments locally.",
        default=None,
    )
    action_group.add_argument(
        "--test",
        "--run-single-slurm",
        action="store_true",
        help="Run the first experiment in list as a test on slurm.",
        default=False,
    )
    action_group.add_argument(
        "--download",
        action="store_true",
        help="download data from wandb from successfull experiments",
        default=False,
    )
    action_group.add_argument(
        "--clear-download",
        action="store_true",
        help="Clear cache of downloaded data from wandb",
        default=False,
    )
    action_group.add_argument(
        "--plot",
        action="store_true",
        help="plot data from experiments",
        default=False,
    )
    action_group.add_argument(
        "--plot-perclass",
        action="store_true",
        help="",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--force-rerun",
        action="store_true",
        help="Force rerun of experiments that are already saved.",
        default=False,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="epoch used for plotting",
        default=None,
    )

    parser.add_argument(
        "--step",
        type=int,
        help="step used for plotting",
        default=None,
    )

    parser.add_argument(
        "--use_step",
        action="store_true",
        help="use steps instead of epochs",
        default=False,
    )

    args = parser.parse_args()

    if args.report:
        report_by_key = args.report if type(args.report) == str else None
        report(experiments, report_by_key)
        return

    if args.single is not None:
        idx = int(args.single)
        if idx < 0 or idx >= len(experiments):
            raise ValueError(
                f"Given index {idx} out of bounds for {len(experiments)} experiments"
            )
        experiments[args.single].run_experiment()
        return

    if args.split_index:
        get_logger().info(f"Preparing to run {len(experiments)} experiments")
        run_split(experiments, split_index=args.split_index, split=args.split_num)
        return

    if args.local or args.slurm:
        get_logger().info(f"Preparing to run {len(experiments)} experiments")
        if args.local:
            run_locally(experiments, force_rerun=args.force_rerun)
        elif args.slurm:
            if slurm_config is None:
                raise ValueError(
                    "Must provide a SlurmConfig if running on slurm. Got None."
                )
            run_slurm(
                experiments,
                slurm_config,
                force_rerun=args.force_rerun,
                python_file=python_file,
            )
        return

    if args.slurm_split:
        get_logger().info(f"Preparing to run {len(experiments)} experiments")
        if slurm_config is None:
            raise ValueError(
                "Must provide a SlurmConfig if running on slurm. Got None."
            )
        run_slurm(
            experiments,
            slurm_config,
            force_rerun=args.force_rerun,
            split=args.slurm_split,
            python_file=python_file,
        )
        return
    if args.test:
        get_logger().info(f"Preparing to run first experiment in group")

        if slurm_config is None:
            raise ValueError(
                "Must provide a SlurmConfig if running on slurm. Got None."
            )
        run_slurm(
            [experiments[0]],
            slurm_config,
            force_rerun=args.force_rerun,
            python_file=python_file,
        )
        return

    if args.download:
        get_logger().info(
            f"Preparing to download data from {len(experiments)} experiments"
        )
        download_data(experiments)
        return

    if args.clear_download:
        get_logger().info(f"Clearing cache for {len(experiments)} experiments")
        clear_downloaded_data(experiments)
        return
    if args.plot:
        get_logger().info(f"Preparing to plot from {len(experiments)} experiments")
        if args.use_step:
            plot_grids(
                experiments=experiments, plotting_time=args.step, using_step=True
            )
            plot_best(experiments=experiments, plotting_time=args.step, using_step=True)
        else:
            plot_grids(experiments=experiments, plotting_time=args.epoch)
            plot_best(experiments=experiments, plotting_time=args.epoch)
        return
    if args.plot_perclass:
        get_logger().info(f"Preparing to plot from {len(experiments)} experiments")

        if not all(
            ["PerClass" in exp.problem.__class__.__name__ for exp in experiments]
        ):
            print("Not a per-class experiment.")
            return
        if not all([experiments[0].group == exp.group for exp in experiments]):
            raise ValueError("Experiments from different groups")
        if not all([experiments[0].problem == exp.problem for exp in experiments]) and not isinstance(experiments[0], LightningExperiment):
            raise ValueError("Experiments on different problems")
        if args.use_step:
            plot_per_class(
                experiments=experiments, plotting_time=args.step, using_step=True
            )
        else:
            plot_per_class(experiments=experiments, plotting_time=args.epoch)
        return

    parser.print_help()


def report(experiments: List[Experiment], by: Optional[str]) -> None:
    """Generate a report of what experiments have been run/are stored on wandb."""
    if by is not None:
        raise NotImplementedError

    all_exps = experiments
    remaining_exps = remove_experiments_that_are_already_saved(experiments)

    n = len(all_exps)
    n_missing = len(remaining_exps)
    percent_complete = ((n - n_missing) / n) * 100
    print(
        f"Out of {n} experiments, {n_missing} still need to run "
        f"({percent_complete:.2f}% complete)"
    )


def remove_experiments_that_are_already_saved(
    experiments: List[Experiment],
) -> List[Experiment]:
    """Checks a list of experiments against the experiments stored on wandb.
    Returns only the experiments that are not saved and marked as successful.

    Args:
        experiments: List of experiments to check

    Returns: List of experiments that still need to run
    """

    if len(experiments) == 0:
        return []

    group = experiments[0].group
    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")

    successful_runs = get_wandb_runs_for_group(group)
    successful_exp_ids = [run.config["exp_id"] for run in successful_runs]

    experiments_to_run = [
        exp for exp in experiments if exp.exp_id() not in successful_exp_ids
    ]

    return experiments_to_run


def run_locally(experiments: List[Experiment], force_rerun: bool) -> None:
    """Run experiments locally."""
    if not force_rerun:
        original_n_exps = len(experiments)
        experiments = remove_experiments_that_are_already_saved(experiments)
        get_logger().info(
            f"New experiments to run: {len(experiments)}/{original_n_exps}"
        )

    datasets = set([exp.problem.dataset for exp in experiments])

    for dataset in datasets:
        if dataset.should_download():
            download_dataset(dataset_name=dataset.name)

    for exp in tqdm(experiments):
        exp.run_experiment()


def run_split(experiments: List[Experiment], split_index: int, split: int):
    exps_to_run = remove_experiments_that_are_already_saved(experiments)

    groups_exps_ind = [
        experiments[n : n + split] for n in range(0, len(experiments), split)
    ]

    exps_split = groups_exps_ind[split_index]

    for exp in exps_split:
        if exp in exps_to_run:
            exp.run_experiment()


def run_slurm(
    experiments: List[Experiment],
    slurm_config: SlurmConfig,
    force_rerun: bool,
    split: Optional[int] = None,
    python_file: Optional[Path] = None,
) -> None:
    """Run experiments on Slurm."""
    print("Preparing experiments to run on Slurm")
    if python_file is None:
        path_to_python_script = Path(sys.argv[0]).resolve()
    else:
        path_to_python_script = python_file

    if not force_rerun:
        print("  Checking which experiments have to run")
        exps_to_run = remove_experiments_that_are_already_saved(experiments)
        should_run = [exp in exps_to_run for exp in experiments]
        print(f"    Should run {should_run.count(True)}/{len(should_run)} experiments")
    else:
        should_run = [True for _ in experiments]

    if not split:
        contents = make_jobarray_file_contents(
            experiment_file=path_to_python_script,
            should_run=should_run,
            slurm_config=slurm_config,
        )
    else:
        contents = make_jobarray_file_contents_split(
            experiment_file=path_to_python_script,
            num_exps=len(experiments),
            slurm_config=slurm_config,
            split=split,
        )

    group = experiments[0].group
    tmp_filename = f"tmp_{group}.sh"
    print(f"  Saving sbatch file in {tmp_filename}")
    with open(tmp_filename, "w+") as file:
        file.writelines(contents)

    datasets = set([exp.problem.dataset for exp in experiments])

    for dataset in datasets:
        if dataset.should_download():
            download_dataset(dataset_name=dataset.name)

    print(f"  Sending experiments to Slurm - executing sbatch file")
    os.system(f"sbatch {tmp_filename}")


def download_data(experiments: List[Experiment]) -> None:
    """Download data from experiments into wandb cache."""

    if len(experiments) == 0:
        return

    group = experiments[0].group

    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")

    if not os.path.exists(config.get_wandb_cache_directory() / group):
        os.makedirs(config.get_wandb_cache_directory() / group)

    successful_run_ids, successful_runs = get_successful_ids_and_runs(group)

    for i, exp in enumerate(experiments):
        if exp.exp_id() not in successful_run_ids:
            get_logger().info(
                f"The following experiment: {str(exp)} (idx {i}) was NOT SUCCESSFULL. NO DATA TO DOWNLOAD."
            )

    runs_to_dl_ids = [exp.exp_id() for exp in experiments]
    runs_to_dl = []
    for run, run_id in zip(successful_runs, successful_run_ids):
        if run_id in runs_to_dl_ids:
            runs_to_dl.append(run)

    for run in tqdm(runs_to_dl):
        download_run_data(run)

    download_summary(group)


def clear_downloaded_data(experiments: List[Experiment]) -> None:
    """Download data from experiments into wandb cache."""

    if len(experiments) == 0:
        return

    group = experiments[0].group

    if not all(exp.group == group for exp in experiments):
        raise ValueError("All experiments must have the same group.")

    path = config.get_wandb_cache_directory() / group
    if not os.path.exists(path):
        return

    rmtree(path)
