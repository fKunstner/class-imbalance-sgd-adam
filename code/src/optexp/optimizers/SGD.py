from dataclasses import dataclass

import torch

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer


@dataclass
class SGD(Optimizer):
    """Wrapper class for defining and loading the SGD optimizer."""

    momentum: float = 0

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(), lr=self.learning_rate.as_float(), momentum=self.momentum
        )


def SGD_M(lr: LearningRate) -> SGD:
    return SGD(learning_rate=lr, momentum=0.9)


def SGD_NM(lr: LearningRate) -> SGD:
    return SGD(learning_rate=lr, momentum=0)
