from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.simpler_transformers.train_only_last_layer_tiny import (
    dataset,
    model,
)
from optexp.optimizers import (
    SGD_M,
    SGD_NM,
    Adam_M,
    Adam_NM,
    NormSGD_M,
    NormSGD_NM,
    Sign_M,
    Sign_NM,
)
from optexp.problems.classification import FullBatchClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config

seeds = [0]

problem = FullBatchClassificationWithPerClassStats(model=model, dataset=dataset)

optimizers = [
    SGD_NM(LearningRate(Fraction(0, 1))),
    SGD_M(LearningRate(Fraction(-1, 2))),
    Adam_NM(LearningRate(Fraction(-2, 1))),
    Adam_M(LearningRate(Fraction(-3, 2))),
    NormSGD_NM(LearningRate(Fraction(0, 1))),
    NormSGD_M(LearningRate(Fraction(-3, 2))),
    Sign_NM(LearningRate(Fraction(-3, 1))),
    Sign_M(LearningRate(Fraction(-7, 2))),
]

group = "MLP_Frozen_100_FB_Start_Config_0.08_split_by_class"

EPOCHS = 1000
experiments = [
    Experiment(optim=opt, problem=problem, group=group, seed=seed, epochs=EPOCHS)
    for opt in optimizers
    for seed in seeds
]

SLURM_CONFIG = slurm_config.SMALL_GPU_4H
if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
