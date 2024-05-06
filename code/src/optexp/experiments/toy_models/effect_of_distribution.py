from optexp import Classification, Experiment, exp_runner_cli
from optexp.datasets.classification_mixture import ClassificationMixture
from optexp.models.linear import LinearInit0
from optexp.optimizers import SGD_NM
from optexp.runner.slurm import slurm_config
from optexp.utils import SEEDS_1, lr_grid

opts = [
    *[SGD_NM(lr) for lr in lr_grid(start=-3, end=3, density=0)],
]


c = 100
alpha = 1
var = 0.5
n = ClassificationMixture.n_samples(c, alpha)
model = LinearInit0(c, c)
dataset = ClassificationMixture(batch_size=n, alpha=1, c=c, var=var)

problem = Classification(model, dataset)
epochs = 1000

group = f"EffectOfDistribution_alpha={alpha}_c={c}_var={var}"

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts, SEEDS_1)],
    problem=problem,
    epochs=epochs,
    group=group,
)

SLURM_CONFIG = slurm_config.SMALL_GPU_4H

if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
