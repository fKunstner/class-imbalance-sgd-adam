from fractions import Fraction

from optexp import Experiment, LearningRate, exp_runner_cli
from optexp.datasets.barcoded_mnist import MNISTAndBarcode
from optexp.models.cnn import SimpleMNISTCNN
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
from optexp.utils import SEEDS_1

EPOCHS = 300
group = "SimpleCNN_MNISTBarcoded_FB_normalized_v2_perclass"

model = SimpleMNISTCNN()
dataset = MNISTAndBarcode(name="MNIST", batch_size=20_000)
problem = FullBatchClassificationWithPerClassStats(model, dataset)

opts = [
    SGD_NM(lr=LearningRate(Fraction(-3, 2))),
    SGD_M(lr=LearningRate(Fraction(-4, 2))),
    Adam_NM(lr=LearningRate(Fraction(-5, 2))),
    Adam_M(lr=LearningRate(Fraction(-5, 2))),
    Sign_NM(lr=LearningRate(Fraction(-4, 2))),
    Sign_M(lr=LearningRate(Fraction(-7, 2))),
    NormSGD_NM(lr=LearningRate(Fraction(0, 2))),
    NormSGD_M(lr=LearningRate(Fraction(0, 2))),
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
