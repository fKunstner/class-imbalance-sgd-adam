from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST

from optexp import config
from optexp.datasets.dataset import Dataset
from optexp.datasets.loaders.batch_loader import BatchLoader

CORNERS_10 = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
)


def generate_combinations(n: int):
    combinations = []
    for i in range(1 << n):
        binary_str = format(i, "0" + str(n) + "b")
        combinations.append(binary_str)
    # remove  the 0 string
    return combinations[1:]


def load_split_mnist(download=False):
    train_set = MNIST(
        root=config.get_dataset_directory(), train=True, download=download
    )

    mnist_x = train_set.data.cpu().numpy()
    mnist_y = train_set.targets.cpu().numpy()
    permutation = np.random.permutation(len(mnist_x))

    mnist_x = mnist_x[permutation]
    mnist_y = mnist_y[permutation]

    tr_va_split = 50_000

    mnist_x_tr = mnist_x[:tr_va_split]
    mnist_y_tr = mnist_y[:tr_va_split]
    mnist_x_va = mnist_x[tr_va_split:]
    mnist_y_va = mnist_y[tr_va_split:]

    return mnist_x_tr, mnist_y_tr, mnist_x_va, mnist_y_va


def make_barcoded_mnist(
    corners=CORNERS_10,
    samples_per_class_tr: int = 5,
    samples_per_class_val: int = 1,
    with_mnist: bool = False,
    normalize: bool = True,
):
    """Creates a barcoded variant of the MNIST dataset.

    Creates 10 * 2^len(corners) classes with samples_per_class samples.

    With default parameters, 10 corners, gives
        10 * 2^10 = 10240 classes
        5 * 10240 = 51200 training samples
        1 * 10240 = 10240 validation samples

    Uses a 50k/10k split for training and validation.
    """
    mnist_x_tr, mnist_y_tr, mnist_x_va, mnist_y_va = load_split_mnist()
    digits = np.unique(mnist_y_tr)
    max_value = mnist_x_tr.max()
    barcodes = product([0, 1], repeat=len(corners))

    assert len(np.unique(mnist_y_tr)) == 10
    assert len(np.unique(mnist_y_va)) == 10

    def apply_barcode(barcode, x):
        img = np.copy(x)
        for binary, corner in zip(barcode, corners):
            if binary == 1:
                img[corner[0]][corner[1]] = max_value
        return img

    def make_new_random_image(candidates, barcode):
        i = np.random.randint(len(candidates))
        return apply_barcode(barcode, candidates[i])

    xs_tr = []
    ys_tr = []
    xs_va = []
    ys_va = []
    class_idx = 0
    for i, barcode in enumerate(barcodes):
        for digit in digits:
            digit_samples_tr = mnist_x_tr[mnist_y_tr == digit]
            digit_samples_va = mnist_x_va[mnist_y_va == digit]
            for j in range(samples_per_class_tr):
                xs_tr.append(make_new_random_image(digit_samples_tr, barcode))
                ys_tr.append(class_idx)
            for j in range(samples_per_class_val):
                xs_va.append(make_new_random_image(digit_samples_va, barcode))
                ys_va.append(class_idx)
            class_idx += 1

    X_tr: Tensor = torch.from_numpy(np.array(xs_tr)).to(torch.float32)
    y_tr: Tensor = torch.from_numpy(np.array(ys_tr)).to(torch.long)
    X_va: Tensor = torch.from_numpy(np.array(xs_va)).to(torch.float32)
    y_va: Tensor = torch.from_numpy(np.array(ys_va)).to(torch.long)

    if with_mnist:
        X_tr = torch.cat([torch.from_numpy(mnist_x_tr), X_tr], dim=0)
        y_tr = torch.cat([torch.from_numpy(mnist_y_tr), y_tr + 10], dim=0)
        X_va = torch.cat([torch.from_numpy(mnist_x_va), X_va], dim=0)
        y_va = torch.cat([torch.from_numpy(mnist_y_va), y_va + 10], dim=0)

    if normalize:

        def safe_normalize(to_normalize, means, stds):
            to_normalize[:, stds > 0] = to_normalize[:, stds > 0] - means[stds > 0]
            to_normalize[:, stds > 0] = to_normalize[:, stds > 0] / stds[stds > 0]
            return to_normalize

        means = X_tr.mean(dim=0)
        stds = X_tr.std(dim=0)
        X_tr = safe_normalize(X_tr, means, stds)
        X_va = safe_normalize(X_va, means, stds)

    X_tr = X_tr.unsqueeze(1)
    X_va = X_va.unsqueeze(1)

    return X_tr, y_tr, X_va, y_va


def make_batch_loader(tr_x, tr_y, va_x, va_y, batch_size, device):
    tr_x = tr_x.to(device)
    tr_y = tr_y.to(device)
    va_x = va_x.to(device)
    va_y = va_y.to(device)
    assert int(tr_y.max()) == int(va_y.max())
    output_shape = np.array([int(tr_y.max()) + 1])
    input_shape = np.array([tr_x.shape[1]])
    train_data = BatchLoader(tr_x, tr_y, batch_size)
    val_data = BatchLoader(va_x, va_y, batch_size)
    return train_data, val_data, input_shape, output_shape, torch.bincount(tr_y)


@dataclass(frozen=True)
class MNISTAndBarcode(Dataset):
    imbalance: bool = True

    def __post_init__(self):
        if not self.imbalance:
            raise ValueError("MNISTImbalanced should always imbalanced")

    def load(self):
        return make_batch_loader(
            *make_barcoded_mnist(
                corners=CORNERS_10,
                samples_per_class_tr=5,
                samples_per_class_val=1,
                with_mnist=True,
            ),
            self.batch_size,
            config.get_device(),
        )


@dataclass(frozen=True)
class MNISTBarcodeOnly(Dataset):
    imbalance: bool = True

    def __post_init__(self):
        if not self.imbalance:
            raise ValueError("MNISTImbalanced should always imbalanced")

    def load(self):
        return make_batch_loader(
            *make_barcoded_mnist(
                corners=CORNERS_10,
                samples_per_class_tr=5,
                samples_per_class_val=1,
                with_mnist=False,
            ),
            self.batch_size,
            config.get_device(),
        )


@dataclass(frozen=True)
class MNISTAndBarcodeNotNormalized(Dataset):
    imbalance: bool = True

    def __post_init__(self):
        if not self.imbalance:
            raise ValueError("MNISTImbalanced should always imbalanced")

    def load(self):
        return make_batch_loader(
            *make_barcoded_mnist(
                corners=CORNERS_10,
                samples_per_class_tr=5,
                samples_per_class_val=1,
                with_mnist=True,
                normalize=False,
            ),
            self.batch_size,
            config.get_device(),
        )


if __name__ == "__main__":
    load_split_mnist(download=True)
