from optexp import (
    Classification,
    Experiment,
    TextDataset,
    TransformerModel,
    exp_runner_cli,
)
from optexp.models.initializer import TransformerEncoderInitializer
from optexp.optimizers import (
    SGD_M,
    SGD_NM,
    Adam_M,
    Adam_NM,
    ScaledSign_M,
    ScaledSign_NM,
    Adagrad,
    LSGD,
)
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_3, lr_grid

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
problem = Classification(model, dataset)

opts_sparse = [
    *[ScaledSign_M(lr) for lr in lr_grid(start=-10, end=-7, density=0)],
    *[ScaledSign_NM(lr) for lr in lr_grid(start=-10, end=-7, density=0)],
]

opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-2, end=0, density=1)],
    *[SGD_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-5, end=-2, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-2, density=1)],
    *[Adagrad(lr) for lr in lr_grid(start=-4, end=-2, density=1)],
    *[ScaledSign_M(lr) for lr in lr_grid(start=-9, end=-7, density=1)],
    *[ScaledSign_NM(lr) for lr in lr_grid(start=-8, end=-6, density=1)],
]

opts = opts_sparse + opts_dense

group = "TEnc_standard_training_PTB"


experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_3)],
    problem=problem,
    epochs=epochs,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_12H


if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
