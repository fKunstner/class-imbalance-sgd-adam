from optexp import MLP, Experiment, exp_runner_cli
from optexp.datasets.synthetic_dataset import BalancedXImbalancedY
from optexp.optimizers import (
    SGD_M,
    SGD_NM,
    Adam_M,
    Adam_NM,
    NormSGD_M,
    NormSGD_NM,
    Sign_M,
    Sign_NM,
    ScaledSign_M,
    ScaledSign_NM,
    LSGD,
    Adagrad,
)
from optexp.problems.classification import FullBatchClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid, starting_grid_for

dataset = BalancedXImbalancedY(batch_size=22_528, size=11)
model = MLP(hidden_layers=None, activation=None)
problem = FullBatchClassification(model, dataset)

group = "LogReg_BalancedXImbalancedY"

opts_sparse = starting_grid_for(
    [SGD_NM, SGD_M, Adam_NM, Adam_M, Sign_M, Sign_NM, NormSGD_M, NormSGD_NM],
    start=-6,
    end=0,
) + [
    *[ScaledSign_M(lr) for lr in lr_grid(start=-10, end=-7, density=0)],
    *[ScaledSign_NM(lr) for lr in lr_grid(start=-9, end=-6, density=0)],
    *[Adagrad(lr) for lr in lr_grid(start=-5, end=-2, density=0)],
    *[LSGD()],
]
opts_dense = [
    *[SGD_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[SGD_NM(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[Sign_M(lr) for lr in lr_grid(start=-6, end=-4, density=1)],
    *[Sign_NM(lr) for lr in lr_grid(start=-6, end=-4, density=1)],
    *[NormSGD_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[NormSGD_NM(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adagrad(lr) for lr in lr_grid(start=-5, end=-2, density=1)],
    *[ScaledSign_M(lr) for lr in lr_grid(start=-9, end=-7, density=1)],
    *[ScaledSign_NM(lr) for lr in lr_grid(start=-9, end=-7, density=1)],
    *[LSGD()],
]

EPOCHS = 1000


experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
