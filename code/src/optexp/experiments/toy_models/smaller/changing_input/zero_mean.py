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

dataset = GaussianImbalancedY(batch_size=896, size=7, x_mean=0.0)
problem = FullBatchClassificationWithPerClassStats(model, dataset)
group = "LogReg_BalancedXImbalancedY_changing_inputs_zero_mean"

opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-4, end=3, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-6, end=1, density=1)],
]

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_dense, SEEDS_1)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
