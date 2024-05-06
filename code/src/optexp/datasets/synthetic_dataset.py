from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from optexp import Dataset, config
from optexp.datasets.loaders.batch_loader import BatchLoader


def generate_dataset(
    m,
    x_generator: Callable = torch.rand,
    add_mean=0.0,
    dim=None,
    add_bias=False,
    scale=1.0,
):
    """Create a synthetic dataset.

    The `size` parameter controls the size of the data and the imbalance.

    The the most and least frequent classes differ in frequency by 2**(size-1)

    The number of samples `n`, classes `c` and dimensionality `d` will be
        n = size * 2**size
        c = 2**size - 1
        d = (size + 1) * 2**size

        size      Memory     Samples   Dimensions    classes     Imbalance
        2       448.0   B         8            12          3             2
        3         3.2 KiB        24            32          7             4
        4        20.5 KiB        64            80         15             8
        5       121.2 KiB       160           192         31            16
        6       675.0 KiB       384           448         63            32
        7         3.5 MiB       896          1024        127            64
        8        18.0 MiB      2048          2304        255           128
        9        90.0 MiB      4608          5120        511           256
        10      440.1 MiB     10240         11264       1023           512
        11        2.1 GiB     22528         24576       2047          1024



    The features X are independent in N(0,1)
    The labels y are imbalanced (step approx. to zipf)
    """
    ys = []
    class_idx = 0
    yfreqs = []
    for i in range(0, m):
        n_classes = 2**i
        n_samples_per_class = 2 ** (m - i)

        for _ in range(n_classes):
            ys.extend([class_idx] * n_samples_per_class)
            yfreqs.append([n_samples_per_class])
            class_idx += 1
    n = len(ys)
    assert n == m * 2**m
    d = (m + 1) * 2**m if dim is None else dim
    y = torch.tensor(ys, dtype=torch.long, device=config.get_device())
    X_raw = x_generator((n, d), dtype=torch.float32, device=config.get_device())
    X = X_raw * scale + add_mean
    if add_bias:
        X = torch.cat((X, torch.ones(n, 1, device=config.get_device())), dim=1)
    return X, y, yfreqs


def make_datalodader(X, y, class_freqs, batch_size):
    num_features = X.shape[1]
    num_classes = len(torch.unique(y))
    train_dataset = BatchLoader(X, y, batch_size)
    val_dataset = BatchLoader(X, y, batch_size)
    class_freqs = torch.tensor(class_freqs, device=X.device).flatten()
    return (
        train_dataset,
        val_dataset,
        np.array([num_features]),
        np.array([num_classes]),
        class_freqs,
    )


@dataclass(frozen=True)
class BalancedXImbalancedY(Dataset):
    batch_size: int
    size: int = 5
    name: str = field(default="balanced_x_imbalanced_y", init=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Size must be greater than 1.")

    def create_data(self):
        return generate_dataset(self.size)

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False


@dataclass(frozen=True)
class GaussianImbalancedY(Dataset):
    batch_size: int
    size: int = 5
    x_mean: float = 0.0
    x_scale: float = 1.0
    name: str = field(default="balanced_x_imbalanced_y_gaussian", init=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Size must be greater than 1.")

    def create_data(self):
        return generate_dataset(
            self.size, x_generator=torch.randn, add_mean=self.x_mean, scale=self.x_scale
        )

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False


@dataclass(frozen=True)
class BalancedXImbalancedY(Dataset):
    batch_size: int
    size: int = 5
    name: str = field(default="balanced_x_imbalanced_y", init=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Size must be greater than 1.")

    def create_data(self):
        return generate_dataset(self.size)

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False


@dataclass(frozen=True)
class GaussianImbalancedYSmallD(Dataset):
    batch_size: int
    size: int = 5
    x_mean: float = 0.0
    dim: int = None
    name: str = field(default="balanced_x_imbalanced_y_gaussian", init=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Size must be greater than 1.")

    def create_data(self):
        return generate_dataset(
            self.size, x_generator=torch.randn, add_mean=self.x_mean, dim=self.dim
        )

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False


@dataclass(frozen=True)
class GaussianImbalancedYBuiltinBias(Dataset):
    batch_size: int
    size: int = 5
    x_mean: float = 0.0
    name: str = field(default="balanced_x_imbalanced_y_gaussian_withbias", init=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Size must be greater than 1.")

    def create_data(self):
        return generate_dataset(
            self.size,
            x_generator=torch.randn,
            add_mean=self.x_mean,
            add_bias=True,
        )

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False
