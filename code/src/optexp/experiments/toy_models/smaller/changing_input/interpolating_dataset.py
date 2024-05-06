from optexp import Experiment, exp_runner_cli
from optexp.datasets.synthetic_dataset import GaussianImbalancedY
from optexp.experiments.toy_models.balanced_x import model
from optexp.experiments.toy_models.smaller.balanced_x_smaller_longer_perclass import (
    EPOCHS,
    opts_dense,
)
from optexp.optimizers import SGD_NM, Adam_NM
from optexp.problems.classification import FullBatchClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, lr_grid

means = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
scales = [1.0, 0.75, 0.5, 0.25, 0.1]

sets_of_experiments = []

opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-4, end=3, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-6, end=1, density=1)],
]

EPOCHS = 500


def make_experiments(mean, scale):
    dataset = GaussianImbalancedY(batch_size=896, size=7, x_mean=mean, x_scale=scale)
    problem = FullBatchClassificationWithPerClassStats(model, dataset)
    group = "LogReg_BalancedXImbalancedY_interp_inputs_mean={}_std={}".format(
        mean, scale
    )
    return Experiment.generate_experiments_from_opts_and_seeds(
        opts_and_seeds=[(opts_dense, SEEDS_1)],
        problem=problem,
        epochs=EPOCHS,
        group=group,
    )


for mean in means:
    sets_of_experiments.append(make_experiments(mean, 1.0))

for scale in scales:
    sets_of_experiments.append(make_experiments(0.0, scale))

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    for experiments in sets_of_experiments:
        exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
