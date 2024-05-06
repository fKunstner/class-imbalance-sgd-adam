from fractions import Fraction
from typing import List

from optexp import (
    Adam,
    AdamW,
    ClassificationWithPerClassStats,
    Experiment,
    LearningRate,
    exp_runner_cli,
)
from optexp.config import UseWandbProject
from optexp.experiments.bigger_models.gpt2small_wt103.general_settings import (
    GROUP_5K,
    get_dataset,
    model,
)
from optexp.experiments.lightning_experiment import LightningExperiment
from optexp.optimizers.SGD import SGD_M
from optexp.runner.slurm import slurm_config

WANDB_PROJECT = "testing"

SLURM_CONFIG = slurm_config.DEFAULT_2_H100_24H

seeds = [0]


steps = 5000
epochs = 0
gradient_acc_steps = 8
train_batch_size = 32
eval_batch_size = 128
target_length = 1024
eval_every = 100

SGD_gradient_acc_steps = 16
SGD_batch_size = 16


dataset = get_dataset(train_batch_size, eval_batch_size, target_length)
dataset_sgd = get_dataset(SGD_batch_size, eval_batch_size, target_length)


problem = ClassificationWithPerClassStats(model, dataset)
sgd_problem = ClassificationWithPerClassStats(model, dataset_sgd)

adamw_lr = LearningRate(exponent=Fraction(-4, 1))
adam_lr = LearningRate(exponent=Fraction(-9, 2))
sgd_lr = LearningRate(exponent=Fraction(-2, 1))

optimizers = [AdamW(adamw_lr, split_decay=True), Adam(adam_lr), SGD_M(sgd_lr)]

experiment_adam = LightningExperiment(
    optim=Adam(adam_lr),
    problem=problem,
    group=GROUP_5K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=gradient_acc_steps,
)
experiment_adamw = LightningExperiment(
    optim=AdamW(adamw_lr, split_decay=True),
    problem=problem,
    group=GROUP_5K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=gradient_acc_steps,
)
experiment_sgd = LightningExperiment(
    optim=SGD_M(sgd_lr),
    problem=sgd_problem,
    group=GROUP_5K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=SGD_gradient_acc_steps,
)

experiments: List[Experiment] = [experiment_sgd, experiment_adam, experiment_adamw]

# experiments = [
#    LightningExperiment(optim=opt, problem=problem, group=group, seed=seed, epochs=epochs, steps=steps, eval_every=eval_every, gradient_acc_steps=gradient_acc_steps)
#    for opt in optimizers
#    for seed in seeds
# ]

if __name__ == "__main__":
    with UseWandbProject(WANDB_PROJECT):
        exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
