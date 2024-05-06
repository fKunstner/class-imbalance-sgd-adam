from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.imbalance.PTB import dataset, model
from optexp.optimizers import (
    LSGD,
    SGD_M,
    SGD_NM,
    Adagrad,
    Adam_M,
    Adam_NM,
    ScaledSign_M,
    ScaledSign_NM,
)
from optexp.problems.classification import ClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1

epochs = 100

problem = ClassificationWithPerClassStats(model, dataset)

optimizers = [
    SGD_NM(LearningRate(exponent=Fraction(-2, 2))),
    SGD_M(LearningRate(exponent=Fraction(-3, 2))),
    # Fraction(-7, 2) is better for Adam, but unstable training.
    # Slightly smaller step-size has very similar end result but stable training.
    Adam_NM(LearningRate(exponent=Fraction(-8, 2))),
    Adam_M(LearningRate(exponent=Fraction(-7, 2))),
]

group = "TEnc_standard_training_PTB_per_class"

experiments = [
    Experiment(optim=opt, problem=problem, group=group, seed=seed, epochs=epochs)
    for opt in optimizers
    for seed in SEEDS_1
]

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
