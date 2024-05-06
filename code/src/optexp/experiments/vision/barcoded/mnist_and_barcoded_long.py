from optexp import Experiment, exp_runner_cli
from optexp.experiments.vision.barcoded.mnist_and_barcoded import (
    opts_dense,
    opts_sparse,
    problem,
)
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, SEEDS_3

EPOCHS = 600
group = "SimpleCNN_MNISTBarcoded_FB_normalized_v2_long"

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.DEFAULT_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
