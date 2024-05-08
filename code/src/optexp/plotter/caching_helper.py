from pathlib import Path

import torch


class CacheFile:
    def __init__(
        self, basedir: str, prefix: str | None = None, suffix: str | None = None
    ):
        self.basedir = Path(basedir)
        self.prefix = "" if prefix is None else prefix + "_"
        self.suffix = "" if suffix is None else "_" + suffix
        if not self.basedir.exists():
            self.basedir.mkdir(parents=True, exist_ok=True)

    def get_path(self, *objs):
        filename = self.prefix + "_".join(map(str, objs)) + self.suffix + ".pt"
        return Path(self.basedir) / filename

    def exists(self, *objs):
        return self.get_path(*objs).exists()

    def save(self, thing, *objs):
        torch.save(thing, self.get_path(*objs))

    def load(self, *objs):
        return torch.load(self.get_path(*objs))


hessian_w_cache = CacheFile("data/hess", suffix="w")
hessian_b_cache = CacheFile("data/hess", suffix="b")
hessian_subset_cache = CacheFile("data/hess_sub", suffix="w")
grad_w_cache = CacheFile("data/grads", suffix="w")
grad_b_cache = CacheFile("data/grads", suffix="b")
x_cache = CacheFile("data", prefix="x")
y_cache = CacheFile("data", prefix="y")
model_cache = CacheFile("data/models", prefix="model")
probs_cache = CacheFile("data/probs", suffix="w")
