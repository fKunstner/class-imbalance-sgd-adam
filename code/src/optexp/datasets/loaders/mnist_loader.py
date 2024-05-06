from typing import Any, Callable, Optional, Tuple

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets.mnist import MNIST


def download_mnist(save_path):
    for train in [True, False]:
        MNISTDataset(save_path, download=True, train=train)


def download_tiny_mnist(save_path):
    download_mnist(save_path)


def load_tiny_mnist(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device
):
    return load_mnist(
        save_path,
        batch_size,
        shuffle,
        num_workers,
        normalize,
        flatten,
        device,
        mode="tiny",
    )


def download_micro_mnist(save_path):
    download_mnist(save_path)


def load_micro_mnist(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device
):
    return load_mnist(
        save_path,
        batch_size,
        shuffle,
        num_workers,
        normalize,
        flatten,
        device,
        mode="micro",
    )


def load_mnist(
    save_path, batch_size, shuffle, num_workers, normalize, flatten, device, mode=None
):
    if normalize:
        mean = torch.tensor(
            0.1307, dtype=torch.float32, device=device, requires_grad=False
        )
        std = torch.tensor(
            0.3081, dtype=torch.float32, device=device, requires_grad=False
        )
    else:
        mean = torch.tensor(0, dtype=torch.float32, device=device)
        std = torch.tensor(1, dtype=torch.float32, device=device)

    if flatten:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda _: _.to(torch.float32)),
                transforms.Lambda(lambda _: (_ - mean) / std),  # normalize inputs
                transforms.Lambda(lambda _: torch.flatten(_)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda _: _.to(torch.float32)),
                transforms.Lambda(lambda _: (_ - mean) / std),  # normalize inputs,
                transforms.Lambda(lambda _: _.unsqueeze(0)),
            ]
        )

    train_set = MNISTDataset(
        save_path,
        download=False,
        train=True,
        transform=transform,
        mode=mode,
    )

    val_set = MNISTDataset(
        save_path,
        download=False,
        train=False,
        transform=transform,
        mode=mode,
    )
    return prepare_loaders(
        num_workers, train_set, batch_size, mode, shuffle, device, val_set
    )


class MNISTDataset(MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.smaller = True if mode else False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if self.smaller:
            img = img.squeeze(0)
            target = target.squeeze(0)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def prepare_loaders(num_workers, train_set, batch_size, mode, shuffle, device, val_set):
    train_set.data = train_set.data.to(device)
    train_set.targets = train_set.targets.to(device)
    val_set.data = val_set.data.to(device)
    val_set.targets = val_set.targets.to(device)
    output_shape = np.array([train_set.targets.max().item() + 1])
    train_set, val_set = subsample(mode, train_set, val_set)
    train_data_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_data_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    loaders = {"train_loader": train_data_loader, "val_loader": val_data_loader}
    features, _ = next(iter(train_data_loader))
    input_shape = np.array(list(features[0].shape))
    return loaders, input_shape, output_shape, torch.bincount(train_set.targets)


def subsample(mode, train_set, val_set):
    if mode:
        train_indices = []
        val_indices = []
        size_per_class_train = 1000 if mode == "tiny" else 500
        size_per_class_val = 100 if mode == "tiny" else 50
        for i in range(0, 10):
            x = torch.eq(train_set.targets, i).nonzero()
            train_indices = train_indices + x[0:size_per_class_train].tolist()
            val_indices = val_indices + x[0:size_per_class_val].tolist()
        train_set = Subset(train_set, indices=train_indices)
        val_set = Subset(val_set, indices=val_indices)
    return train_set, val_set
