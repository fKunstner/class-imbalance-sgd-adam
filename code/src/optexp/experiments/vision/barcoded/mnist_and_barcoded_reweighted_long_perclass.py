from optexp import Experiment, exp_runner_cli, LearningRate
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
from optexp.models.cnn import SimpleMNISTCNN
from optexp.datasets.barcoded_mnist import MNISTAndBarcode
from fractions import Fraction
from optexp.problems.classification import FullBatchWeightedClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3

EPOCHS = 600
group = "SimpleCNN_MNISTBarcoded_FB_normalized_v2_reweighted_long_perclass"

model = SimpleMNISTCNN()
dataset = MNISTAndBarcode(name="MNIST", batch_size=20_000)
problem = FullBatchWeightedClassificationWithPerClassStats(model, dataset)


opts = [
    SGD_M(LearningRate(exponent=Fraction(-1, 2))),
    SGD_NM(LearningRate(exponent=Fraction(0, 1))),
    Adam_M(LearningRate(exponent=Fraction(-5, 2))),
    Adam_NM(LearningRate(exponent=Fraction(-5, 2))),
    NormSGD_M(LearningRate(exponent=Fraction(-2, 1))),
    NormSGD_NM(LearningRate(exponent=Fraction(0, 1))),
    Sign_M(LearningRate(exponent=Fraction(-4, 1))),
    Sign_NM(LearningRate(exponent=Fraction(-5, 2)))
]


experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_1)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.DEFAULT_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
