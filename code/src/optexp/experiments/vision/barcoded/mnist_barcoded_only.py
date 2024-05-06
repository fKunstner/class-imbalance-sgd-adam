from optexp import Experiment, exp_runner_cli
from optexp.datasets.barcoded_mnist import MNISTBarcodeOnly
from optexp.models.cnn import SimpleMNISTCNN
from optexp.optimizers import SGD_M, SGD_NM, Adam_M, Adam_NM
from optexp.problems.classification import FullBatchClassification
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_3, lr_grid, starting_grid_for

EPOCHS = 300
group = "SimpleCNN_MNISTBarcodedOnly_FB_normalized"

model = SimpleMNISTCNN()
# dataset = MNISTBarcodeOnly(name="MNISTBalancedBarcoded", batch_size=20_000)
# dataset = MNISTBalancedBarcoded(name="MNISTBalancedBarcoded", batch_size=512) # debugging
# problem = Classification(model, dataset) # debugging
dataset = MNISTBarcodeOnly(name="MNIST", batch_size=20_000)
problem = FullBatchClassification(model, dataset)


opts_sparse = starting_grid_for(
    [SGD_NM, SGD_M, Adam_NM, Adam_M],
    start=-5,
    end=0,
)


opts_dense = [
    *[SGD_NM(lr) for lr in lr_grid(start=-1, end=2, density=1)],
    *[SGD_M(lr) for lr in lr_grid(start=-2, end=2, density=1)],
    *[Adam_NM(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
    *[Adam_M(lr) for lr in lr_grid(start=-3, end=-1, density=1)],
]

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_3), (opts_dense, SEEDS_3)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = slurm_config.DEFAULT_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
