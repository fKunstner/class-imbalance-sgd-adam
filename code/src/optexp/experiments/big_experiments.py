from pathlib import Path

from optexp import exp_runner_cli
from optexp.config import UseWandbProject


def dispatch_cmdline_to_all_groups(groups):
    for group in groups:
        if group.__file__ is None:
            raise ValueError(f"Module {group.__name__} has no __file__ attribute")
        print(f"Module {group.__name__} at {group.__file__}")

        if hasattr(group, "WANDB_PROJECT"):
            with UseWandbProject(group.WANDB_PROJECT):
                exp_runner_cli(
                    experiments=group.experiments,
                    slurm_config=group.SLURM_CONFIG,
                    python_file=Path(group.__file__),
                )
        else:
            exp_runner_cli(
                experiments=group.experiments,
                slurm_config=group.SLURM_CONFIG,
                python_file=Path(group.__file__),
            )


if __name__ == "__main__":

    from optexp.experiments.bigger_models.gpt2small_wt103 import (
        gpt2small_wt103,
        gpt2small_wt103_with_class_stats,
        gpt2small_wt103_with_class_stats_long,
    )

    groups = [
        gpt2small_wt103,
        gpt2small_wt103_with_class_stats,
        gpt2small_wt103_with_class_stats_long,
    ]

    dispatch_cmdline_to_all_groups(groups)
