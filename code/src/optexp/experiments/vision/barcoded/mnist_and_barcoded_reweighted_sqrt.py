from optexp import Experiment, exp_runner_cli
from optexp.datasets.barcoded_mnist import MNISTAndBarcode
from optexp.models.cnn import SimpleMNISTCNN
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import FullBatchWeightedClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_3, starting_grid_for

EPOCHS = 500
group = "SimpleCNN_MNISTBarcoded_FB_normalized_v2_reweighted_sqrt"

model = SimpleMNISTCNN()
dataset = MNISTAndBarcode(name="MNIST", batch_size=20_000)
problem = FullBatchWeightedClassification(model, dataset, sqrt_weights=True)

opts_sparse = starting_grid_for(
    [
        SGD_NM,
        SGD_M,
        Adam_NM,
        Adam_M,
    ],
    start=-7,
    end=2,
)

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.DEFAULT_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
