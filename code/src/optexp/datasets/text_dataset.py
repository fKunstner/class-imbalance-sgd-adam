from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from optexp.config import get_logger
from optexp.datasets import Dataset, MixedBatchSizeDataset, get_text_dataset


@dataclass(frozen=True)
class TextDataset(Dataset):
    """A Text Dataset

    Attributes:
        tgt_len: length of sequence to be inputted into model

    """

    tgt_len: int = 35

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_text_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            tgt_len=self.tgt_len,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )


@dataclass(frozen=True)
class MixedBatchSizeTextDataset(MixedBatchSizeDataset):
    """A Text Dataset

    Attributes:
        tgt_len: length of sequence to be inputted into model

    """

    batch_size = None  # type: ignore
    tgt_len: int = 1024

    def load(self):
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_text_dataset(
            dataset_name=self.name,
            batch_size=self.train_batch_size,
            tgt_len=self.tgt_len,
            mixed_batch_size=True,
            eval_batch_size=self.eval_batch_size,
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


@dataclass(frozen=True)
class VariableTokenTextDataset(TextDataset):
    merge_factor: int = 1

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_text_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            tgt_len=self.tgt_len,
            merge=self.merge_factor,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )
