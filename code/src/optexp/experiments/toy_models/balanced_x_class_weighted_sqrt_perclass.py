from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.toy_models.balanced_x_class_weighted_sqrt import problem
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1

group = "LogReg_ClassWeighted_BalancedXImbalancedY_sqrt_perclass"

opts = [
    SGD_NM(lr=LearningRate(Fraction(-3, 2))),
    SGD_M(lr=LearningRate(Fraction(-2, 1))),
    Adam_NM(lr=LearningRate(Fraction(-9, 2))),
    Adam_M(lr=LearningRate(Fraction(-4, 1))),
]

EPOCHS = 500

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_1)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)


SLURM_CONFIG = slurm_config.SMALL_GPU_3H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
