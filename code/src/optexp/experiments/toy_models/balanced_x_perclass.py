from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.toy_models.balanced_x import dataset, model
from optexp.optimizers import (
    LSGD,
    SGD_M,
    SGD_NM,
    Adagrad,
    Adam_M,
    Adam_NM,
    NormSGD_M,
    NormSGD_NM,
    ScaledSign_M,
    ScaledSign_NM,
    Sign_M,
    Sign_NM,
)
from optexp.problems.classification import FullBatchClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config

seeds = [0]
problem = FullBatchClassificationWithPerClassStats(model, dataset)
group = "LogReg_BalancedXImbalancedY_PerClass"


optimizers = [
    # Frac(-2,1) is better at the end but unstable throughout
    SGD_NM(LearningRate(Fraction(-5, 2))),
    # Frac(-5, 2) is better at the end but unstable first half
    SGD_M(LearningRate(Fraction(-3, 1))),
    Adam_NM(LearningRate(Fraction(-4, 1))),
    Adam_M(LearningRate(Fraction(-7, 2))),
    # Adagrad(LearningRate(Fraction(-3, 1))),
    # LSGD(),
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
