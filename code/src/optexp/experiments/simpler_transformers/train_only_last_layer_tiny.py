from optexp import (
    MLP,
    Experiment,
    FrozenDataset,
    FullBatchClassification,
    exp_runner_cli,
)
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
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid, starting_grid_for

seeds = [0, 1, 2]

dataset = FrozenDataset(name="Frozen_100", batch_size=80_000, porp=0.08)
model = MLP(hidden_layers=None, activation=None)

problem = FullBatchClassification(model=model, dataset=dataset)


opts_sparse = starting_grid_for(
    [SGD_NM, SGD_M, Adam_NM, Adam_M, Sign_M, Sign_NM, NormSGD_M, NormSGD_NM],
    start=-5,
    end=1,
)
opts_dense = [
    *[SGD_M(lr) for lr in lr_grid(start=-1, end=1, density=1)],
    *[SGD_NM(lr) for lr in lr_grid(start=-1, end=1, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[NormSGD_M(lr) for lr in lr_grid(start=-2, end=0, density=1)],
    *[NormSGD_NM(lr) for lr in lr_grid(start=-1, end=1, density=1)],
    *[Sign_M(lr) for lr in lr_grid(start=-4, end=-2, density=1)],
    *[Sign_NM(lr) for lr in lr_grid(start=-4, end=-2, density=1)],
]

group = "MLP_Frozen_100_FB_Start_Config_0.08"

EPOCHS = 3200

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_HALF

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
