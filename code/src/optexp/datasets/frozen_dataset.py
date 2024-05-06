from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from optexp.config import get_logger
from optexp.datasets import Dataset, get_frozen_dataset


@dataclass(frozen=True)
class FrozenDataset(Dataset):
    """Frozen Dataset generated by freezing the embedding and attention layers of Transformer

    Attributes:
        porp: the porportion of the original dataset to take
    """

    porp: float = 1.0

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_frozen_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            porp=self.porp,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )