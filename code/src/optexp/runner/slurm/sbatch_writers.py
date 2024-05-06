"""Module to integrate with Slurm."""

import math
import textwrap
from pathlib import Path
from typing import List

from optexp import config
from optexp.runner.slurm.slurm_config import SlurmConfig


def make_sbatch_header(slurm_config: SlurmConfig, n_jobs: int) -> str:
    """Generates the header of a sbatch file for Slurm.

    Args:
        slurm_config: Slurm configuration to use
        n_jobs: Number of jobs to run in the batch
    """

    gpu_str = ""
    if slurm_config.gpu is not None:
        if isinstance(slurm_config.gpu, str):
            gpu_str = f"#SBATCH --gpus-per-node={slurm_config.gpu}:{slurm_config.n_gpu}"
        elif isinstance(slurm_config.gpu, bool):
            if slurm_config.gpu:
                gpu_str = f"#SBATCH --gpus-per-node={slurm_config.n_gpu}"
            else:
                gpu_str = ""

    bangline = "#!/bin/sh\n"

    formatted_header = textwrap.dedent(
        """
        #SBATCH --account={acc}
        #SBATCH --mem={mem}
        #SBATCH --time={time}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mail-user={email}
        #SBATCH --mail-type=ALL
        #SBATCH --array=0-{last_job_idx}
        {gpu_str}

        """
    ).format(
        acc=config.get_slurm_account(),
        mem=slurm_config.mem,
        time=slurm_config.time,
        cpus=slurm_config.cpus,
        email=config.get_slurm_email(),
        gpu_str=gpu_str,
        last_job_idx=n_jobs - 1,
    )

    return bangline + formatted_header


def make_jobarray_content(
    run_exp_by_idx_command: str,
    should_run: List[bool],
):
    """Creates the content of a jobarray sbatch file for Slurm.

    Args:
        run_exp_by_idx_command: Command to call to run the i-th experiment
        should_run: Whether the matching experiment should run
    """

    bash_script_idx_to_exp_script_idx = []
    for i, _should_run in enumerate(should_run):
        if _should_run:
            bash_script_idx_to_exp_script_idx.append(i)

    commands_for_each_experiment = []
    for bash_script_idx, exp_script_idx in enumerate(bash_script_idx_to_exp_script_idx):
        commands_for_each_experiment.append(
            textwrap.dedent(
                f"""
                if [ $SLURM_ARRAY_TASK_ID -eq {bash_script_idx} ]
                then
                    {run_exp_by_idx_command} {exp_script_idx}
                fi
                """
            )
        )

    return "".join(commands_for_each_experiment)


def make_jobarray_content_split(
    run_exp_by_idx_command: str, num_splits: int, split: int
):
    """Creates the content of a jobarray sbatch file for Slurm."""

    # bash_script_idx_to_exp_script_idx = []

    commands_for_each_experiment = []
    for i in range(num_splits):
        commands_for_each_experiment.append(
            textwrap.dedent(
                f"""
                if [ $SLURM_ARRAY_TASK_ID -eq {i} ]
                then
                    {run_exp_by_idx_command} --split_index {i} --split_num  {split}
                fi
                """
            )
        )

    return "".join(commands_for_each_experiment)


def make_jobarray_file_contents(
    experiment_file: Path,
    should_run: List[bool],
    slurm_config: SlurmConfig,
):
    """Creates a jobarray sbatch file for Slurm."""

    header = make_sbatch_header(slurm_config=slurm_config, n_jobs=sum(should_run))

    body = make_jobarray_content(
        run_exp_by_idx_command=f"python {experiment_file} --single",
        should_run=should_run,
    )

    footer = textwrap.dedent(
        """
        exit
        """
    )

    return header + body + footer


def make_jobarray_file_contents_split(
    experiment_file: Path,
    num_exps: int,
    slurm_config: SlurmConfig,
    split: int,
):
    """Creates a jobarray sbatch file for Slurm."""

    num_splits = math.ceil(num_exps / split)

    header = make_sbatch_header(slurm_config=slurm_config, n_jobs=num_splits)

    body = make_jobarray_content_split(
        run_exp_by_idx_command=f"python {experiment_file}",
        num_splits=num_splits,
        split=split,
    )

    footer = textwrap.dedent(
        """
        exit
        """
    )

    return header + body + footer
