from dataclasses import dataclass

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from torchvision.datasets import MNIST

from optexp import Dataset, config
from optexp.datasets.loaders.batch_loader import BatchLoader

BASE_CLASSES = [i for i in range(0, 10)]


class MNISTDataset(MNIST):
    pass


def create_balanced_coloured_dataset(
    batch_size, device, many_colours=False, return_loader=True, normalize=True
):

    val_split = 50000
    if many_colours:
        n_hue = 50
        n_sat = 20
        n_per_train_class = 5
        n_per_val_class = 1
    else:
        n_hue = 20  # 50
        n_sat = 5  # 20
        n_per_train_class = 50
        n_per_val_class = 10

    n_colours = n_hue * n_sat

    train_set = MNISTDataset(
        root=config.get_dataset_directory(), train=True, download=False
    )

    # give three channels
    data = train_set.data.cpu().numpy()
    targets = train_set.targets.cpu().numpy()
    data = np.expand_dims(data, axis=-1)
    data = np.concatenate([data, data, data], axis=-1) / 255.0

    # do split
    n = targets.size
    shuffel = np.random.permutation(n)
    data, targets = data[shuffel], targets[shuffel]
    val_data = data[val_split:]
    val_targets = targets[val_split:]
    train_data = data[:val_split]
    train_targets = targets[:val_split]

    # colour
    saturation_vals = np.linspace(0, 1, n_sat + 1)[1:]
    hue_vals = np.linspace(0, 1, n_hue + 1)[1:]
    hue_sat_pairs = np.transpose(
        [
            np.tile(hue_vals, len(saturation_vals)),
            np.repeat(saturation_vals, len(hue_vals)),
        ]
    )  # 1000x2
    hue_sat_pairs = np.transpose(
        np.expand_dims(hue_sat_pairs, axis=(-1, -2)), axes=(0, 3, 2, 1)
    )  # 1000x1x1x2
    hue_sat_pairs = np.repeat(hue_sat_pairs, 28, axis=2)  # 1000x1x28x2
    image_hue_sat_pairs = np.repeat(hue_sat_pairs, 28, axis=1)  # 1000x28x28x2

    new_train_data = []
    new_train_targets = []

    new_train_target_idx = 0
    for base_class in BASE_CLASSES:
        base_classes = train_data[train_targets == base_class]
        for colour_idx in range(n_colours):
            base_classes = base_classes[
                np.random.choice(
                    base_classes.shape[0], size=n_per_train_class, replace=False
                )
            ]
            for i in range(n_per_train_class):
                x = base_classes[i]
                x[:, :, 0:2] = image_hue_sat_pairs[colour_idx]
                x = hsv_to_rgb(x)
                x = np.transpose(x, axes=(2, 0, 1))
                new_train_data.append(x)
                new_train_targets.append(new_train_target_idx)
            new_train_target_idx += 1

    new_val_data = []
    new_val_targets = []

    new_val_target_idx = 0
    for base_class in BASE_CLASSES:
        base_classes = val_data[val_targets == base_class]
        for colour_idx in range(n_colours):
            base_classes = base_classes[
                np.random.choice(
                    base_classes.shape[0], size=n_per_val_class, replace=False
                )
            ]
            for i in range(n_per_val_class):
                x = base_classes[i]
                x[:, :, 0:2] = image_hue_sat_pairs[colour_idx]
                x = hsv_to_rgb(x)
                x = np.transpose(x, axes=(2, 0, 1))
                new_val_data.append(x)
                new_val_targets.append(new_val_target_idx)
            new_val_target_idx += 1

    train_data = np.array(new_train_data)
    train_targets = np.array(new_train_targets)
    val_data = np.array(new_val_data)
    val_targets = np.array(new_val_targets)

    shuffel = np.random.permutation(train_targets.size)
    train_data, train_targets = train_data[shuffel], train_targets[shuffel]

    if not return_loader:
        return train_data, train_targets, val_data, val_targets

    if normalize:
        train_data_means = np.mean(train_data, axis=0)
        train_data_stds = np.std(train_data, axis=0) + 10e-16
        train_data = (train_data - train_data_means) / train_data_stds

        val_data = (val_data - train_data_means) / train_data_stds
        val_data[:, train_data_stds <= 10e-15] = 0.0

    train_data = torch.from_numpy(train_data).to(torch.float32).to(device)
    train_targets = torch.from_numpy(train_targets).to(torch.long).to(device)

    val_data = torch.from_numpy(val_data).to(torch.float32).to(device)
    val_targets = torch.from_numpy(val_targets).to(torch.long).to(device)

    output_shape = np.array([int(train_targets.max()) + 1])
    input_shape = np.array([train_data.shape[1]])

    train_data = BatchLoader(train_data, train_targets, batch_size)
    val_data = BatchLoader(val_data, val_targets, batch_size)

    return (
        train_data,
        val_data,
        input_shape,
        output_shape,
        torch.bincount(train_targets),
    )


def create_unbalanced_coloured_dataset(batch_size, device, many_colours=False):

    val_split = 50000

    train_set = MNISTDataset(
        root=config.get_dataset_directory(), train=True, download=False
    )

    data = train_set.data.cpu().numpy()
    targets = train_set.targets.cpu().numpy()

    data = train_set.data.cpu().numpy()
    targets = train_set.targets.cpu().numpy()
    data = np.expand_dims(data, axis=-1)
    data = np.concatenate([data, data, data], axis=-1) / 255.0

    n = targets.size
    shuffel = np.random.permutation(n)
    data, targets = data[shuffel], targets[shuffel]
    val_data = data[val_split:]
    val_targets = targets[val_split:]
    train_data = data[:val_split]
    train_targets = targets[:val_split]

    sparse_train_data, sparse_train_targets, sparse_val_data, sparse_val_targets = (
        create_balanced_coloured_dataset(
            batch_size, device, many_colours=many_colours, return_loader=False
        )
    )

    train_data = np.transpose(train_data, axes=(0, 3, 1, 2))
    val_data = np.transpose(val_data, axes=(0, 3, 1, 2))

    train_data = np.concatenate([train_data, sparse_train_data])
    train_targets = np.concatenate([train_targets, sparse_train_targets + 10])
    val_data = np.concatenate([val_data, sparse_val_data])
    val_targets = np.concatenate([val_targets, sparse_val_targets + 10])

    train_data_means = np.mean(train_data, axis=0)
    train_data_stds = np.std(train_data, axis=0) + 10e-16
    train_data = (train_data - train_data_means) / train_data_stds

    val_data = (val_data - train_data_means) / train_data_stds
    val_data[:, train_data_stds <= 10e-15] = 0.0
    
    n_train = train_targets.size
    shuffel = np.random.permutation(n_train)
    train_data, train_targets = train_data[shuffel], train_targets[shuffel]

    n_val = val_targets.size
    shuffel = np.random.permutation(n_val)
    val_data, val_targets = val_data[shuffel], val_targets[shuffel]


    train_data = torch.from_numpy(train_data).to(torch.float32).to(device)
    train_targets = torch.from_numpy(train_targets).to(torch.long).to(device)

    val_data = torch.from_numpy(val_data).to(torch.float32).to(device)
    val_targets = torch.from_numpy(val_targets).to(torch.long).to(device)
    
    

    output_shape = np.array([int(train_targets.max()) + 1])
    input_shape = np.array([train_data.shape[1]])

    train_data = BatchLoader(train_data, train_targets, batch_size)
    val_data = BatchLoader(val_data, val_targets, batch_size)

    return (
        train_data,
        val_data,
        input_shape,
        output_shape,
        torch.bincount(train_targets),
    )



@dataclass(frozen=True)
class MNISTColouredNotNormalized(Dataset):
    """
    MNIST with balanced classes but lots of them

    """

    def load(self):
        return create_balanced_coloured_dataset(self.batch_size, config.get_device(), normalize=False)

    def should_download(self):
        return False


@dataclass(frozen=True)
class MNISTUnbalancedColoured(Dataset):
    """
    MNIST with balanced classes but lots of them

    """

    def load(self):
        return create_unbalanced_coloured_dataset(self.batch_size, config.get_device())

    def should_download(self):
        return False


@dataclass(frozen=True)
class MNISTUnbalancedManyColoured(Dataset):
    """
    MNIST with balanced classes but lots of them

    """

    def load(self):
        return create_unbalanced_coloured_dataset(
            self.batch_size, config.get_device(), many_colours=True
        )

    def should_download(self):
        return False


@dataclass(frozen=True)
class MNISTBalancedManyColoured(Dataset):
    """
    MNIST with balanced classes but lots of them

    """

    def load(self):
        return create_balanced_coloured_dataset(
            self.batch_size, config.get_device(), many_colours=True
        )

    def should_download(self):
        return False


@dataclass(frozen=True)
class MNISTBalancedColoured(Dataset):
    """
    MNIST with balanced classes but lots of them

    """

    def load(self):
        return create_balanced_coloured_dataset(self.batch_size, config.get_device())

    def should_download(self):
        return False
