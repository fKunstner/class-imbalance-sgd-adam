from dataclasses import dataclass

import torch

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer


@dataclass
class Adam(Optimizer):
    """Wrapper class for defining and loading the Adam optimizer."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate.as_float(),
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )


def Adam_NM(lr: LearningRate) -> Adam:
    return Adam(learning_rate=lr, beta1=0)


def Adam_M(lr: LearningRate) -> Adam:
    return Adam(learning_rate=lr, beta1=0.9)
