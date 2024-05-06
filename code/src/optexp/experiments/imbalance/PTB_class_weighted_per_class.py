from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.experiments.imbalance.PTB_class_weighted import dataset, model
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import WeightedClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1

epochs = 100

problem = WeightedClassificationWithPerClassStats(model, dataset, sqrt_weights=True)

opts = [
    SGD_NM(LearningRate(exponent=Fraction(0, 1))),
    SGD_M(LearningRate(exponent=Fraction(-1, 2))),
    Adam_NM(LearningRate(exponent=Fraction(-7, 2))),
    Adam_M(LearningRate(exponent=Fraction(-7, 2))),
]

group = "TEnc_standard_training_class_weighted_PTB_per_class"

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_1)],
    problem=problem,
    epochs=epochs,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_6H


if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
