from fractions import Fraction
from typing import Callable, List

import numpy as np

from optexp.config import get_logger
from optexp.experiments.experiment import Experiment
from optexp.optimizers import SGD, Adam, LearningRate, Optimizer


def _check_parameters_logspace(end, start, density):
    if density < 0 or not np.allclose(int(density), density):
        raise ValueError(f"Density needs to be an integer >= 0, got {density}.")
    if not np.allclose(int(start), start) or not np.allclose(int(end), end):
        raise ValueError(f"Start and end need to be integers, got {start, end}.")
    assert end > start


def nice_logspace(start, end, base, density):
    """Returns a log-spaced grid between ``base**start`` and ``base**end``.

    Plays nicely with ``merge_grids``. Increasing the density repeats previous values.

    - ``Start``, ``end`` and ``density`` are integers.
    - Increasing ``density`` by 1 doubles the number of points
    - ``Density = 1`` will return ``end - start + 1`` points
    - ``Density = 2`` will return ``2*(end-start) + 1`` points
    - ``nice_logspace(0, 1, base=10, density=1) == [1, 10] == [10**0, 10**1]``
    - ``nice_logspace(0, 1, base=10, density=2) == [1, 3.16..., 10] == [10**0, 10**(1/2), 10**1]``
    """
    _check_parameters_logspace(end, start, density)
    return np.logspace(start, end, base=base, num=(end - start) * (2**density) + 1)


def lr_grid(start, end, density=0, base=10):
    """Returns the values of the nice_logspace function as LearningRates."""
    return nice_learning_rates(start, end, base, density)


def nice_learning_rates(start, end, base, density):
    """Returns the values of the nice_logspace function as LearningRates."""
    _check_parameters_logspace(end, start, density)

    num = (end - start) * (2**density)
    den = 2**density

    exponents = [Fraction(i + den * start, den) for i in range(0, num + 1)]
    learning_rates = [
        LearningRate(exponent=exponent, base=base) for exponent in exponents
    ]
    return learning_rates


def merge_grids(*grids):
    """Merge two lists of parameters.

    Given lists [a,b,c], [c,d,e], returns [a,b,c,d,e]
    """
    return sorted(list(set.union(*[set(grid) for grid in grids])))


def starting_grid_for(
    optimizers: List[Callable[[LearningRate], Optimizer]], start: int, end: int
):
    lrs = lr_grid(start=start, end=end, base=10, density=0)
    return [opt(lr) for opt in optimizers for lr in lrs]


def get_sparse_optimizers(start: int, end: int):
    """Gets Adam + SGD optimizers with and w/o momentum using a sparse grid of
    learning rates (density 0)

    Args:
        start: starting learning rate
        end: final learning rate

    Returns:
        List[Optimizer]: A list of optimizers
    """
    learning_rates = lr_grid(start=start, end=end, base=10, density=0)
    sgd_no_mom_optims: List[Optimizer] = [SGD(lr) for lr in learning_rates]
    sgd_mom_optims: List[Optimizer] = [SGD(lr, momentum=0.9) for lr in learning_rates]
    adam_no_mom_optims: List[Optimizer] = [Adam(lr, beta1=0.0) for lr in learning_rates]
    adam_mom_optims: List[Optimizer] = [Adam(lr) for lr in learning_rates]

    optimizers: List[Optimizer] = sum(
        [sgd_no_mom_optims, sgd_mom_optims, adam_no_mom_optims, adam_mom_optims], []
    )

    return optimizers


def remove_duplicate_exps(experiments: List[Experiment]):
    exps = []
    for exp in experiments:
        if exp not in exps:
            exps.append(exp)
    if len(experiments) > len(exps):
        num_dups = len(experiments) - len(exps)
        get_logger().info(
            f"# unique experiments = {len(exps)} ({len(experiments)} exps, {num_dups} duplicates)"
        )
    return exps


def split_classes_by_frequency(x, n_splits=2):
    """Given an array `x` of size `N` from `C = len(np.unique(x))` classes,

    splits `unique_classes = np.unique(x)` into `n_splits` splits such that
    - The first split contains the most frequent classes
    - ...
    - The last split contains the least frequent classes
    - Each split contains ~ `N / n_splits` samples in `x`,
        np.sum([np.sum(x == c) for c in split]) ~ N / n_splits
    """

    unique_labels, labels_count = np.unique(x, return_counts=True)
    n_labels = len(unique_labels)
    sort_idx = np.flip(labels_count.argsort())
    sorted_labels = unique_labels[sort_idx]
    freq_sorted = labels_count[sort_idx] / labels_count.sum()

    cum_freq_sorted = freq_sorted.cumsum()
    freq_breakpoints = np.linspace(0, 1, n_splits, endpoint=False)[1:]
    indices = np.searchsorted(cum_freq_sorted, freq_breakpoints, side="left")
    splits = np.split(sorted_labels, indices + 1)

    n_labels_in_group = sum(map(len, splits))
    if n_labels_in_group != n_labels:
        raise ValueError(
            f"Class split did not work? "
            f"Expected {n_labels} classes, but got {n_labels_in_group}"
        )

    return splits


SEEDS_1 = [0]
SEEDS_3 = [0, 1, 2]
