from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.imbalance.PTB_with_class_stats import epochs, group, problem
from optexp.optimizers import LSGD, Adagrad, ScaledSign_M, ScaledSign_NM
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1

optimizers = [
    ScaledSign_M(LearningRate(exponent=Fraction(-15, 2))),
    ScaledSign_NM(LearningRate(exponent=Fraction(-15, 2))),
    Adagrad(LearningRate(exponent=Fraction(-5, 2))),
    LSGD(),
]

experiments = [
    Experiment(optim=opt, problem=problem, group=group, seed=seed, epochs=epochs)
    for opt in optimizers
    for seed in SEEDS_1
]

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
