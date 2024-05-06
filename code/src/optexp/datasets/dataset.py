from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from optexp.config import get_logger
from optexp.datasets import get_dataset


@dataclass(frozen=True)
class Dataset:
    """
    Defining and loading the dataset.

    Attributes:
        name: The name of the dataset to load.
        batch_size: The batch size to use.

    """

    name: str
    batch_size: int

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Uses a helper function to load a dataset with PyTorch DataLoader class.

        Returns:
            First PyTorch DataLoader object corresponds to the train set and second is the validation set.
            First NumPy arrays represents the input shape (shape of the features) and the second
            is output shape (shape of the labels) which is useful for defining the model.
        """

        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            split_prop=0.8,
            shuffle=True,
            num_workers=0,
            normalize=True,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )

    def should_download(self):
        return True


@dataclass(frozen=True)
class MixedBatchSizeDataset(Dataset):
    """
    Defining and loading the dataset with different batch size for evaluation and training.

    Attributes:
        name: The name of the dataset to load.
        train_batch_size: The batch size to use in training.
        eval_batch_size: The batch size to use in evaluation.

    """

    name: str
    train_batch_size: int
    eval_batch_size: int

    def load(self):
        """
        Uses a helper function to load a dataset with PyTorch DataLoader class.

        Returns:
            First PyTorch DataLoader object corresponds to the train set and second is the validation set.
            First NumPy arrays represents the input shape (shape of the features) and the second
            is output shape (shape of the labels) which is useful for defining the model.
        """

        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            split_prop=0.8,
            shuffle=True,
            num_workers=0,
            normalize=True,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            loaders["eval_train_loader"],
            loaders["eval_val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )

    def should_download(self):
        return True


@dataclass(frozen=True)
class DummyDataset(Dataset):
    name: str = ""
    batch_size: int = 0

    def load(self):
        raise NotImplementedError("Dummy dataset not intended for use")
