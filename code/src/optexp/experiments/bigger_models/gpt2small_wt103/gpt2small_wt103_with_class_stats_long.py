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
    GROUP_15K,
    get_dataset,
    model,
)
from optexp.experiments.lightning_experiment import LightningExperiment
from optexp.optimizers.SGD import SGD_M
from optexp.runner.slurm import slurm_config

WANDB_PROJECT = "testing"

SLURM_CONFIG = slurm_config.DEFAULT_2_H100_24H

seeds = [0]

steps = 15000
epochs = 0

sgd_gradient_acc_steps = 8
adam_gradient_acc_steps = 16
adamw_gradient_acc_steps = 8

sgd_train_batch_size = 16
adam_train_batch_size = 16
adamw_train_batch_size = 32

eval_batch_size = 128
target_length = 1024
eval_every = 100


sgd_dataset = get_dataset(sgd_train_batch_size, eval_batch_size, target_length)
adam_dataset = get_dataset(adam_train_batch_size, eval_batch_size, target_length)
adamw_dataset = get_dataset(adamw_train_batch_size, eval_batch_size, target_length)

sgd_problem = ClassificationWithPerClassStats(model, sgd_dataset)
adam_problem = ClassificationWithPerClassStats(model, adam_dataset)
adamw_problem = ClassificationWithPerClassStats(model, adamw_dataset)

adamw_lr = LearningRate(exponent=Fraction(-4, 1))
adam_lr = LearningRate(exponent=Fraction(-9, 2))
sgd_lr = LearningRate(exponent=Fraction(-2, 1))

optimizers = [AdamW(adamw_lr, split_decay=True), Adam(adam_lr), SGD_M(sgd_lr)]


experiment_sgd = LightningExperiment(
    optim=SGD_M(sgd_lr),
    problem=sgd_problem,
    group=GROUP_15K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=sgd_gradient_acc_steps,
)
experiment_adam = LightningExperiment(
    optim=Adam(adam_lr),
    problem=adam_problem,
    group=GROUP_15K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=adam_gradient_acc_steps,
)
experiment_adamw = LightningExperiment(
    optim=AdamW(adamw_lr, split_decay=True),
    problem=adamw_problem,
    group=GROUP_15K,
    seed=0,
    epochs=epochs,
    steps=steps,
    eval_every=eval_every,
    gradient_acc_steps=adamw_gradient_acc_steps,
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
