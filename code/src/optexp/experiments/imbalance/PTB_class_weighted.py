from optexp import (
    Experiment,
    TextDataset,
    TransformerModel,
    exp_runner_cli,
)
from optexp.models.initializer import TransformerEncoderInitializer
from optexp.problems.classification import WeightedClassification
from optexp.optimizers import (
    SGD_M,
    SGD_NM,
    Adam_M,
    Adam_NM,
)
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3, lr_grid

epochs = 100

dataset = TextDataset(name="PTB", batch_size=512, tgt_len=35)
model = TransformerModel(
    num_heads=4,
    depth=2,
    width_mlp=1000,
    emb_dim=1000,
    drop_out=0.2,
    init=TransformerEncoderInitializer.default(),
)
problem = WeightedClassification(model, dataset, sqrt_weights=True)

opts_sparse = [
    *[SGD_NM(lr) for lr in lr_grid(start=-3, end=0, density=0)],
    *[SGD_M(lr) for lr in lr_grid(start=-3, end=0, density=0)],
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=-2, density=0)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-2, density=0)],
]

opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-1, end=1, density=1)],
    *[SGD_M(lr) for lr in lr_grid(start=-2, end=0, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
]

group = "TEnc_standard_training_class_weighted_PTB"


experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=epochs,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_6H


if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
