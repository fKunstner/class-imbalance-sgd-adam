from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.toy_models.balanced_x_perclass import group, problem, seeds
from optexp.optimizers import (
    NormSGD_M,
    NormSGD_NM,
    ScaledSign_M,
    ScaledSign_NM,
    Sign_M,
    Sign_NM,
)
from optexp.runner.slurm import slurm_config

optimizers = [
    NormSGD_NM(LearningRate(Fraction(-1, 1))),
    NormSGD_M(LearningRate(Fraction(-3, 2))),
    Sign_NM(LearningRate(Fraction(-5, 1))),
    Sign_M(LearningRate(Fraction(-5, 1))),
    ScaledSign_M(LearningRate(Fraction(-8, 1))),
    ScaledSign_NM(LearningRate(Fraction(-8, 1))),
]

experiments = [
    Experiment(
        optim=opt,
        problem=problem,
        group=group,
        seed=seed,
        epochs=1000,
    )
    for opt in optimizers
    for seed in seeds
]

SLURM_CONFIG = slurm_config.SMALL_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
