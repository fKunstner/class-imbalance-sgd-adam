from optexp import Experiment, exp_runner_cli
from optexp.datasets.synthetic_dataset import GaussianImbalancedYBuiltinBias
from optexp.experiments.toy_models.balanced_x import model
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import FullBatchClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, lr_grid, starting_grid_for

dataset = GaussianImbalancedYBuiltinBias(batch_size=896, size=7)
problem = FullBatchClassification(model, dataset)
group = "LogReg_BalancedXImbalancedY_smaller_longer_withbias"

opts_sparse = starting_grid_for([SGD_NM, SGD_M, Adam_NM, Adam_M], start=-6, end=3)
grid = lr_grid(start=-4, end=2, density=1)
opts_dense = [
    *[SGD_M(lr) for lr in grid],
    *[SGD_NM(lr) for lr in grid],
    *[Adam_M(lr) for lr in grid],
    *[Adam_NM(lr) for lr in grid],
]
EPOCHS = 1000
experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1)],  # , (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
