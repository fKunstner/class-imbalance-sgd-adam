import math
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from optexp.problems import Problem
from optexp.problems.utils import DivergingException


@dataclass
class FullBatchProblem(Problem):
    """Wrapper for a model and dataset defining a problem to optimize."""

    def init_problem(self) -> None:
        super().init_problem()
        self.num_train_samples = self.get_num_samples(val=False)
        self.num_val_samples = self.get_num_samples(val=True)

    def one_epoch(self, optimizer: torch.optim.Optimizer) -> dict:
        """
        Optimizes the model on a specific loss function defined for this problem
        for one epoch on the training set.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters

        Raises:
            DivergingException: Raised when the value for any of the metrics is NAN or INF.

        Returns:
            Keys are strings representing the name of the criterions and values
            are the accumulated values of the criterion on the entire training dataset.
        """
        needs_closure = "lr" not in optimizer.defaults
        grad_norms: Dict[str, List] = {}
        t = time.time()

        def closure(compute_grad=True):
            overall_loss = 0.0
            total_weight = 0
            for _, (features, labels) in enumerate(self.train_loader):
                y_pred = self.torch_model(features)

                output = self.criterion(y_pred, labels)

                if type(output) is not tuple:
                    loss = output
                    weight = len(features) / self.num_train_samples
                    loss *= weight
                    total_weight += weight
                else:
                    loss, weight = output
                    total_weight += weight

                if math.isnan(loss) or math.isinf(loss):
                    raise DivergingException("Live training loss is NAN or INF.")

                overall_loss += loss
                if compute_grad:
                    loss.backward()

            # Divide by total weight

            for p in self.torch_model.parameters():
                if p.grad is not None:
                    p.grad /= total_weight

            overall_loss /= total_weight

            return overall_loss

        loss = closure()
        self._grad_norms(grad_norms_dict=grad_norms)
        optimizer.step(
            closure=lambda: closure(compute_grad=False) if needs_closure else None
        )
        optimizer.zero_grad()

        param_norms = self._param_norms()
        metrics = {
            "live_train_loss": loss.item(),
            "time_per_epoch": time.time() - t,
        }

        metrics.update(param_norms)
        metrics.update(grad_norms)
        return metrics

    def get_num_samples(self, val: bool):
        num_samples = 0
        loader = self.val_loader if val else self.train_loader
        for _, (features, labels) in enumerate(loader):
            num_samples += len(features)
        return num_samples
