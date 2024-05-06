from fractions import Fraction
from typing import List

from optexp import (
    Adam,
    AdamW,
    Classification,
    Experiment,
    LearningRate,
    MixedBatchSizeTextDataset,
    exp_runner_cli,
)
from optexp.config import UseWandbProject
from optexp.experiments.bigger_models.gpt2small_wt103.general_settings import model
from optexp.experiments.lightning_experiment import LightningExperiment
from optexp.optimizers.SGD import SGD_M
from optexp.runner.slurm import slurm_config
from optexp.utils import merge_grids
from optexp.utils import nice_learning_rates as nice_lrs

WANDB_PROJECT = "testing"

seeds = [0, 1]

depth = 12
num_heads = 12
emb_dim = 768

steps = 5000
epochs = 0
target_length = 1024


sgd_gradient_acc_steps_1gpu = 16
sgd_gradient_acc_steps_2gpu = 8
sgd_train_batch_size = 32
sgd_eval_batch_size = 128
sgd_eval_every = 100

adam_gradient_acc_steps = 16
adam_train_batch_size = 8
adam_eval_batch_size = 40
adam_eval_every = 50


sgd_dataset = MixedBatchSizeTextDataset(
    name="WikiText103",
    train_batch_size=sgd_train_batch_size,
    eval_batch_size=sgd_eval_batch_size,
    tgt_len=target_length,
    batch_size=None,  # type: ignore
)

adam_dataset = MixedBatchSizeTextDataset(
    name="WikiText103",
    train_batch_size=adam_train_batch_size,
    eval_batch_size=adam_eval_batch_size,
    tgt_len=target_length,
    batch_size=None,  # type: ignore
)

adam_problem = Classification(model, adam_dataset)
sgd_problem = Classification(model, sgd_dataset)

base_lrs = nice_lrs(start=-5, end=0, base=10, density=0)

adam_lrs = merge_grids(base_lrs, nice_lrs(start=-7, end=-4, base=10, density=1))
sgd_lrs = merge_grids(nice_lrs(start=-4, end=-1, base=10, density=1))

adamw_optimizers = [AdamW(lr, split_decay=True) for lr in adam_lrs]
adam_optimizers = [Adam(lr) for lr in adam_lrs]

sgd_1gpu_lrs = [
    LearningRate(exponent=Fraction(-1, 1)),
    LearningRate(exponent=Fraction(-2, 1)),
    LearningRate(exponent=Fraction(-3, 2)),
]

sgd_2gpu_lrs = [
    LearningRate(exponent=Fraction(-3, 1)),
    LearningRate(exponent=Fraction(-4, 1)),
    LearningRate(exponent=Fraction(-5, 2)),
    LearningRate(exponent=Fraction(-7, 2)),
]

sgd_1gpu_optimizers = [SGD_M(lr) for lr in sgd_1gpu_lrs]
sgd_2gpu_optimizers = [SGD_M(lr) for lr in sgd_2gpu_lrs]

group = "gpt_test"


sgd_1gpu_experiments: List[Experiment] = [
    LightningExperiment(
        optim=opt,
        problem=sgd_problem,
        group=group,
        seed=seed,
        epochs=epochs,
        steps=steps,
        eval_every=sgd_eval_every,
        gradient_acc_steps=sgd_gradient_acc_steps_1gpu,
    )
    for opt in sgd_1gpu_optimizers
    for seed in seeds
]

sgd_2gpu_experiments: List[Experiment] = [
    LightningExperiment(
        optim=opt,
        problem=sgd_problem,
        group=group,
        seed=seed,
        epochs=epochs,
        steps=steps,
        eval_every=sgd_eval_every,
        gradient_acc_steps=sgd_gradient_acc_steps_2gpu,
    )
    for opt in sgd_2gpu_optimizers
    for seed in seeds
]

adam_experiments: List[Experiment] = [
    LightningExperiment(
        optim=opt,
        problem=adam_problem,
        group=group,
        seed=seed,
        epochs=epochs,
        steps=steps,
        eval_every=adam_eval_every,
        gradient_acc_steps=adam_gradient_acc_steps,
    )
    for opt in adam_optimizers
    for seed in seeds
    if not (seed == 1 and opt.learning_rate.exponent == 0)
]

adamw_experiments: List[Experiment] = [
    LightningExperiment(
        optim=opt,
        problem=adam_problem,
        group=group,
        seed=seed,
        epochs=epochs,
        steps=steps,
        eval_every=adam_eval_every,
        gradient_acc_steps=adam_gradient_acc_steps,
    )
    for opt in adamw_optimizers
    for seed in seeds
]


# there should be 57 total  experiments (one of the adam seeds never ran)

experiments: List[Experiment] = (
    sgd_1gpu_experiments + sgd_2gpu_experiments + adam_experiments + adamw_experiments
)

SLURM_CONFIG = slurm_config.DEFAULT_2_A100_48H

if __name__ == "__main__":
    with UseWandbProject(WANDB_PROJECT):
        exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
