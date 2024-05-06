from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from optexp.optimizers.learning_rate import LearningRate


@dataclass
class Optimizer(ABC):
    """Abstract base class for definition of optimizers."""

    learning_rate: LearningRate

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Loads an optimizer with the model parameters

        Args:
            model (torch.nn.Module): The model to optimize

        Returns:
            torch.optim.Optimizer: Torch optimizer to use for training
        """
        pass
