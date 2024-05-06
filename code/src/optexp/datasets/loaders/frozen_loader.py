import numpy as np
import torch

from optexp.datasets.loaders.batch_loader import BatchLoader


def load_frozen_600(save_path, batch_size, device, porp=1.0):
    return load_frozen(save_path, batch_size, device, amount=600, porp=porp)


def load_frozen_300(save_path, batch_size, device, porp=1.0):
    return load_frozen(save_path, batch_size, device, amount=300, porp=porp)


def load_frozen_100(save_path, batch_size, device, porp=1.0):
    return load_frozen(save_path, batch_size, device, amount=100, porp=porp)


def load_frozen(save_path, batch_size, device, amount=1200, porp=1.0):
    if amount == 1200:
        save_path = save_path / "Frozen_1200"
    elif amount == 600:
        save_path = save_path / "Frozen_600"
    elif amount == 300:
        save_path = save_path / "Frozen_300"
    elif amount == 100:
        save_path = save_path / "Frozen_100"

    x_train = np.load(save_path / "train_data.npy")
    y_train = np.load(save_path / "train_targets.npy")

    x_val = np.load(save_path / "val_data.npy")
    y_val = np.load(save_path / "val_targets.npy")

    num_train_samples = int(x_train.shape[0] * porp)
    num_val_samples = int(x_val.shape[0] * porp)

    x_train = torch.tensor(x_train[0:num_train_samples]).to(torch.float32).to(device)
    y_train = torch.tensor(y_train[0:num_train_samples]).to(torch.long).to(device)

    x_val = torch.tensor(x_val[0:num_val_samples]).to(torch.float32).to(device)
    y_val = torch.tensor(y_val[0:num_val_samples]).to(torch.long).to(device)

    train_loader = BatchLoader(x_train, y_train, batch_size)
    val_loader = BatchLoader(x_val, y_val, batch_size)

    output_shape = np.array([int(y_train.max()) + 1])
    input_shape = np.array([x_val.shape[1]])

    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    return loaders, input_shape, output_shape, torch.bincount(y_train)
