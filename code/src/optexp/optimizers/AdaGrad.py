from dataclasses import dataclass
import torch

from optexp.optimizers.optimizer import Optimizer


@dataclass
class Adagrad(Optimizer):
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adagrad(model.parameters(), lr=self.learning_rate.as_float())
