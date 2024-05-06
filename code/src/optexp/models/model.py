from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass()
class Model(ABC):
    """Abstract base class for defining models"""

    @abstractmethod
    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> torch.nn.Module:
        """Loads a PyTorch model given the input features and outputs

        Args:
            input_shape: Shape of features that model will take as input
            output_shape: Shape of predictions that model should output
        """
        pass


@dataclass()
class DummyModel(Model):
    def load_model(self, input_shape, output_shape):
        raise NotImplementedError("DummyModel is not meant to be loaded")
