from optexp import MLP, Experiment, exp_runner_cli
from optexp.datasets.synthetic_dataset import BalancedXImbalancedY
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import FullBatchClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_3, lr_grid

dataset = BalancedXImbalancedY(batch_size=22_528, size=11)
model = MLP(hidden_layers=None, activation=None)
problem = FullBatchClassification(model, dataset)

group = "LogReg_BalancedXImbalancedY_largescale"

opts_dense = [
    *[SGD_M(lr) for lr in lr_grid(start=-5, end=5, density=2)],
    *[SGD_NM(lr) for lr in lr_grid(start=-5, end=5, density=2)],
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=5, density=2)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=5, density=2)],
]
EPOCHS = 10_000

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_12H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
