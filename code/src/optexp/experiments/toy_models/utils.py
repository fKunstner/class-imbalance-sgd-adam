import math
from dataclasses import dataclass

from optexp import Dataset


def create_dataset(start_examples: int, base: int = 10):
    num_groups = math.log(start_examples, base) + 1

    if num_groups.is_integer():
        num_groups = int(num_groups)
    else:
        Exception(f"number of starting examples has to be a power of the base")
    num_groups = round(num_groups)
    class_count = 0
    groups = []
    for i in range(num_groups):
        num_classes_in_group = int(start_examples / (base**i))
        num_examples_per_class_in_group = base**i

        classes = np.array(
            [i for i in range(class_count, class_count + num_classes_in_group)]
        )
        data_for_group = np.tile(classes, num_examples_per_class_in_group)
        groups.append(data_for_group)
        class_count += num_classes_in_group

    y = np.stack(groups).flatten()
    X = np.eye(class_count)[y]

    return X, y, class_count


def load_data(batch_size: int, start_examples: int, base: int = 10):
    X, y, num_classes = create_dataset(start_examples, base=base)

    X_train = torch.tensor(X).to(torch.float32).to(config.get_device())
    y_train = torch.tensor(y).to(torch.long).to(config.get_device())

    X_val = X_train
    y_val = y_train

    train_dataset = BatchLoader(X_train, y_train, batch_size)
    val_dataset = BatchLoader(X_val, y_val, batch_size)

    num_features = num_classes

    return (
        train_dataset,
        val_dataset,
        np.array([num_features]),
        np.array([num_classes]),
        torch.bincount(y_train),
    )


@dataclass(frozen=True)
class DummyLinReg(Dataset):
    start_examples: int
    base: int = 10

    def load(self):
        return load_data(self.batch_size, self.start_examples, self.base)


from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_classification

from optexp import Dataset, config
from optexp.datasets.loaders.batch_loader import BatchLoader


def create_zipf_dummy(
    num_samples: int,
    num_features: int,
    num_classes: int,
    zipf_constant: float,
    batch_size: int,
    porp_informative: float = 0.1,
):
    ranks = np.arange(1, num_classes + 1)
    freqs = 1.0 / ((ranks + 2.7) ** zipf_constant)
    freqs = freqs / np.sum(freqs)

    X, y = make_classification(
        n_samples=int(num_samples * 1.1),
        n_features=num_features,
        n_redundant=0,
        n_informative=int(porp_informative * num_features),
        n_classes=num_classes,
        n_clusters_per_class=1,
        shuffle=False,
        weights=freqs,
        class_sep=2.0,
        flip_y=0.0,
    )

    X_train = X[:num_samples]
    y_train = y[:num_samples]

    X_val = X[num_samples:]
    y_val = y[num_samples:]

    X_train = torch.tensor(X_train).to(torch.float32).to(config.get_device())
    y_train = torch.tensor(y_train).to(torch.long).to(config.get_device())

    X_val = torch.tensor(X_val).to(torch.float32).to(config.get_device())
    y_val = torch.tensor(y_val).to(torch.long).to(config.get_device())

    train_dataset = BatchLoader(X_train, y_train, batch_size)
    val_dataset = BatchLoader(X_val, y_val, batch_size)

    return (
        train_dataset,
        val_dataset,
        np.array([num_features]),
        np.array([num_classes]),
    )


@dataclass(frozen=True)
class SyntheticZipfDataset(Dataset):
    num_samples: int
    num_features: int
    num_classes: int
    zipf_constant: float
    porp_informative: float

    def load(self):
        return create_zipf_dummy(
            self.num_samples,
            self.num_features,
            self.num_classes,
            self.zipf_constant,
            self.batch_size,
            self.porp_informative,
        )
