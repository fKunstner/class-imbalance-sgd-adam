from optexp import Experiment, exp_runner_cli
from optexp.experiments.toy_models.balanced_x_class_weighted import dataset, model
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import FullBatchWeightedClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid, starting_grid_for

problem = FullBatchWeightedClassification(model, dataset, sqrt_weights=True)

group = "LogReg_ClassWeighted_BalancedXImbalancedY_sqrt"

opts_sparse = starting_grid_for(
    [SGD_NM, SGD_M, Adam_NM, Adam_M],
    start=-6,
    end=0,
)

opts_dense = [
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[SGD_M(lr) for lr in lr_grid(start=-3, end=0, density=1)],
    *[SGD_NM(lr) for lr in lr_grid(start=-2, end=1, density=1)],
]

EPOCHS = 1000

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_3H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
