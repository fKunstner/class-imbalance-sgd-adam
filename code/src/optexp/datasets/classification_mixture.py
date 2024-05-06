from dataclasses import dataclass, field

import numpy as np
import torch

from optexp import Dataset, config
from optexp.datasets.synthetic_dataset import make_datalodader


@dataclass(frozen=True)
class ClassificationMixture(Dataset):
    """Creates a classification dataset where the data comes from a mixture of Gaussian.

    The distribution of the classes is a power law
    with power alpha (class k has frequency 𝜋ₖ ∝ 1/k^𝛼).

    The inputs are generated from a mixture of c Gaussians in c dimensions,
    where the mean of the kth Gaussian is kth basis vector and the covariance is the identity.

    The number of samples is set such that the least frequent class has min_n sample.

    Parameters:

    """

    batch_size: int
    alpha: float = 1
    c: int = 100
    min_n: int = 10
    var: float = 0.1
    name: str = field(default="balanced_x_imbalanced_y", init=False)

    @classmethod
    def n_samples(cls, c, alpha):
        return int(np.sum(cls.__n_samples_per_class(c, alpha)))

    @classmethod
    def __n_samples_per_class(cls, c, alpha):
        n_samples_per_class_propto = np.array([1 / k**alpha for k in range(1, c + 1)])
        min_propto = n_samples_per_class_propto[-1]
        approx_n_samples_per_class = n_samples_per_class_propto / min_propto
        approx_n_samples_per_class *= cls.min_n
        n_samples_per_class = np.ceil(approx_n_samples_per_class).astype(int)
        return n_samples_per_class

    def create_data(self):
        n_samples_per_class = self.__n_samples_per_class(self.c, self.alpha)
        freqs = n_samples_per_class / np.sum(n_samples_per_class)
        y = np.concatenate([np.full(n, i) for i, n in enumerate(n_samples_per_class)])
        X = np.random.randn(len(y), self.c) * self.var
        for i in range(self.c):
            X[y == i, i] += 1

        X_t = torch.tensor(X, dtype=torch.float32, device=config.get_device())
        y_t = torch.tensor(y, dtype=torch.long, device=config.get_device())
        freqs_t = torch.tensor(freqs, dtype=torch.float32, device=config.get_device())

        return X_t, y_t, freqs_t

    def load(self):
        X, y, class_freqs = self.create_data()
        return make_datalodader(X, y, class_freqs, self.batch_size)

    def should_download(self):
        return False
