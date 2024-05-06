from optexp import (
    Experiment,
    FullBatchClassification,
    TextDataset,
    TransformerEncoderInitializer,
    TransformerModel,
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

EPOCHS = 3200

dataset = TextDataset(name="TinyPTB", batch_size=1024, tgt_len=35)

avs_initializer = TransformerEncoderInitializer.default()

model = TransformerModel(
    num_heads=1,
    depth=1,
    width_mlp=1200,
    emb_dim=1200,
    drop_out=0.2,
    init=avs_initializer,
)

problem = FullBatchClassification(model, dataset)

opts_sparse = starting_grid_for(
    [SGD_NM, SGD_M, Adam_NM, Adam_M, Sign_M, Sign_NM, NormSGD_M, NormSGD_NM],
    start=-6,
    end=1,
)

opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-2, end=0, density=1)],
    *[SGD_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-4, end=-2, density=1)],
    *[Sign_NM(lr) for lr in lr_grid(start=-4, end=-2, density=1)],
    *[Sign_M(lr) for lr in lr_grid(start=-5, end=-3, density=1)],
    *[NormSGD_NM(lr) for lr in lr_grid(start=-1, end=1, density=1)],
    *[NormSGD_M(lr) for lr in lr_grid(start=-2, end=0, density=1)],
]

group = "TransformerEncoder_TinyPTB_FB_Width_EmbDim_1200"


experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)


SLURM_CONFIG = slurm_config.DEFAULT_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
