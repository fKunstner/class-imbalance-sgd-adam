from typing import List

import numpy as np
import pandas as pd
import torch


def normalize_y_axis(*axes):
    miny, maxy = np.inf, -np.inf
    for ax in axes:
        y1, y2 = ax.get_ylim()
        miny = np.min([miny, y1])
        maxy = np.max([maxy, y2])
    for ax in axes:
        ax.set_ylim([miny, maxy])


def normalize_x_axis(*axes):
    minx, maxx = np.inf, -np.inf
    for ax in axes:
        x1, x2 = ax.get_xlim()
        minx = np.min([minx, x1])
        maxx = np.max([maxx, x2])
    for ax in axes:
        ax.set_xlim([minx, maxx])


def subsample(xs, NMAX=200, log_only=False, linear_only=False):
    if isinstance(xs, torch.Tensor):
        xs = xs.numpy()
    if isinstance(xs, List):
        xs = np.array(xs)
    if isinstance(xs, pd.Series):
        xs = xs.values

    if not isinstance(xs, np.ndarray):
        import pdb

        pdb.set_trace()

    n = len(xs)
    if n < NMAX:
        return xs

    idx_lin = np.floor(np.linspace(0, n, int(NMAX / 2), endpoint=False)).astype(int)
    idx_log = np.floor(
        np.logspace(0, np.log10(n), int(NMAX / 2), endpoint=False)
    ).astype(int)
    base = [0, n - 1]
    indices = set(base)

    if log_only and linear_only:
        raise ValueError("Cannot have both log_only and linear_only")

    if not log_only:
        indices = indices.union(set(idx_lin))
    if not linear_only:
        indices = indices.union(set(idx_log))

    idx = np.array(sorted(list(indices))).astype(int)

    if any(idx >= n):
        import pdb

        pdb.set_trace()

    return xs[idx]
