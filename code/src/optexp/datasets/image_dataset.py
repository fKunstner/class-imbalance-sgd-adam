from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from optexp.config import get_logger
from optexp.datasets.dataset import Dataset
from optexp.datasets.dataset_getter import get_image_dataset


@dataclass(frozen=True)
class ImageDataset(Dataset):
    """An Image Dataset

    Attributes:
        flatten: 2D images will be flattened into a 1D vector if true
    """

    flatten: bool = False

    def load(
        self,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, torch.Tensor]:
        get_logger().info("Loading dataset: " + self.name)

        loaders, input_shape, output_shape, class_freqs = get_image_dataset(
            dataset_name=self.name,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            normalize=True,
            flatten=self.flatten,
        )

        return (
            loaders["train_loader"],
            loaders["val_loader"],
            input_shape,
            output_shape,
            class_freqs,
        )


@dataclass(frozen=True)
class MNIST(ImageDataset):
    batch_size: int
    name: str = field(default="MNIST", init=False)
    flatten: bool = False


@dataclass(frozen=True)
class ImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="ImageNet", init=False)
    flatten: bool = False


@dataclass(frozen=True)
class ImbalancedImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="ImbalancedImageNet", init=False)
    flatten: bool = False



@dataclass(frozen=True)
class TenBigClassImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="TenBigClassImageNet", init=False)
    flatten: bool = False



@dataclass(frozen=True)
class OneMajorClassImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="OneMajorClassImageNet", init=False)
    flatten: bool = False

@dataclass(frozen=True)
class SmallImageNet(ImageDataset):
    batch_size: int
    name: str = field(default="SmallImageNet", init=False)
    flatten: bool = False
