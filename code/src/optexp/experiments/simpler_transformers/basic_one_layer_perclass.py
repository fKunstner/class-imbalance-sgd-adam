from fractions import Fraction

from optexp import (
    Experiment,
    LearningRate,
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
from optexp.problems.classification import FullBatchClassificationWithPerClassStats
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1

EPOCHS = 500

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

problem = FullBatchClassificationWithPerClassStats(model, dataset)

opts = [
    SGD_NM(lr=LearningRate(Fraction(-2, 2))),
    SGD_M(lr=LearningRate(Fraction(-3, 2))),
    Adam_NM(lr=LearningRate(Fraction(-8, 2))),
    Adam_M(lr=LearningRate(Fraction(-7, 2))),
    NormSGD_NM(lr=LearningRate(Fraction(0, 2))),
    NormSGD_M(lr=LearningRate(Fraction(-2, 2))),
    Sign_NM(lr=LearningRate(Fraction(-7, 2))),
    Sign_M(lr=LearningRate(Fraction(-10, 2))),
]

group = "TransformerEncoder_TinyPTB_FB_Width_EmbDim_1200_perclass_smaller"

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_1)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.DEFAULT_GPU_8H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
