import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch

from optexp.config import get_device, get_logger
from optexp.datasets import Dataset, MixedBatchSizeDataset
from optexp.models import Model
from optexp.problems.utils import DivergingException


@dataclass
class Problem(ABC):
    """Wrapper for a model and dataset defining a problem to optimize.

    Attributes:
        model: The model that will be optimized.
        dataset: The dataset to use.
    """

    model: Model
    dataset: Dataset

    def init_problem(self) -> None:
        """Loads the dataset and the PyTorch model onto device."""
        get_logger().info("Loading problem: " + self.__class__.__name__)

        if isinstance(self.dataset, MixedBatchSizeDataset):
            self.is_mixed_batch = True
            (
                self.train_loader,
                self.val_loader,
                self.eval_train_loader,
                self.eval_val_loader,
                self.input_shape,
                self.output_shape,
                self.class_freqs,
            ) = self.dataset.load()
        else:
            self.is_mixed_batch = False
            (
                self.train_loader,
                self.val_loader,
                self.input_shape,
                self.output_shape,
                self.class_freqs,
            ) = self.dataset.load()
        self.torch_model = self.model.load_model(
            self.input_shape, self.output_shape
        ).to(get_device())
        self.criterion = self.init_loss()

    def eval(self, val: bool = True, return_raw: bool = False) -> dict:
        """Wrapper to evaluate model. Provides the list of criterions to use.

        Args:
            val (bool, optional): When True model is evaluated on validation dataset,
                otherwise training dataset. Defaults to True.

        Returns:
            A dictionary where keys are strings representing the name of the criterions and values
            are the accumulated values of the criterions on the entire dataset.
        """
        return self._evaluate(self.get_criterions(), val, return_raw)

    def one_epoch(self, optimizer: torch.optim.Optimizer) -> dict:
        """Optimizes the model on a specific loss function defined for this
        problem for one epoch on the training set.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters

        Raises:
            DivergingException: Raised when the value for any of the metrics is NAN or INF.

        Returns:
            Keys are strings representing the name of the criterions and values
            are the accumulated values of the criterion on the entire training dataset.
        """
        needs_closure = "lr" not in optimizer.defaults
        train_loss = 0.0
        num_samples = 0
        mini_batch_losses = []
        grad_norms: Dict[str, List] = {}
        for _, (features, labels) in enumerate(self.train_loader):

            def closure():
                y_pred = self.torch_model(features)
                out = self.criterion(y_pred, labels)

                if type(out) is tuple:
                    loss, weight = out
                    loss = loss / weight
                else:
                    loss = out

                if math.isnan(loss) or math.isinf(loss):
                    raise DivergingException("Live training loss is NAN or INF.")

                return loss

            optimizer.zero_grad()

            loss = closure()
            loss.backward()
            optimizer.step(closure=closure if needs_closure else None)

            self._grad_norms(grad_norms_dict=grad_norms)
            mini_batch_losses.append(loss.item())
            train_loss += loss.item() * len(features)
            num_samples += len(features)

        param_norms = self._param_norms()
        metrics = {
            "live_train_loss": train_loss / num_samples,
            "mini_batch_losses": mini_batch_losses,
        }

        metrics.update(param_norms)
        metrics.update(grad_norms)
        return metrics

    def _evaluate(
        self,
        criterions: List[torch.nn.Module],
        val: bool = True,
        return_raw: bool = False,
    ) -> dict:
        """Evaluates the model on the dataset on the given criterions without
        any optimization.

        Args:
            criterions (List[torch.nn.Module]): A list of loss functions and other metrics used for evaluation.
            val (bool, optional): When True model is evaluated on validation dataset,
                otherwise training dataset. Defaults to True.

        Raises:
            DivergingException: Raised when the value for any of the metrics is NAN or INF.

        Returns:
            A dictionary where keys are strings representing the name of the criterions and values
                are the accumulated values of the criterions on the entire dataset.
        """
        num_samples = 0
        running_metrics: Dict[torch.nn.Module, torch.Tensor] = {}
        running_n_samples: Dict[torch.nn.Module, torch.Tensor] = {}
        

        def add_(d, k, v):
            if k in d:
                d[k] += v
            else:
                d[k] = v
            return d

        if val:
            loader = self.eval_val_loader if self.is_mixed_batch else self.val_loader
            key_prefix = "va"
        else:
            loader = (
                self.eval_train_loader if self.is_mixed_batch else self.train_loader
            )
            key_prefix = "tr"

        with torch.no_grad():
            for _, (features, labels) in enumerate(loader):
                y_pred = self.torch_model(features)
                for module in criterions:
                    outputs = module(y_pred, labels)
                    if type(outputs) is not tuple:
                        value = outputs.detach()

                        if math.isnan(value) or math.isinf(value):
                            raise DivergingException(
                                f"{key_prefix}_{str(module)[:-2]} is NAN or INF."
                            )
                        add_(running_metrics, module, value * len(features))
                    else:
                        value, weight = outputs[0].detach(), outputs[1].detach()
                        add_(running_metrics, module, value)
                        add_(running_n_samples, module, weight)
                
                num_samples += len(features)
                
                del y_pred
                del features
                del labels
        if return_raw:
            return {
                "metric_values": running_metrics,
                "metric_n_samples": running_n_samples,
                "total_samples": num_samples,
            }

        else:
            for module in criterions:
                if torch.numel(running_metrics[module]) > 1:
                    running_metrics[module] = (
                        running_metrics[module] / running_n_samples[module]
                    )
                else:
                    running_metrics[module] /= num_samples

            metrics = {
                f"{key_prefix}_{str(module)[:-2]}": (
                    value.tolist() if torch.numel(value) > 1 else value.item()
                )
                for module, value in running_metrics.items()
            }

            return metrics

    def _grad_norms(self, grad_norms_dict: dict) -> None:
        """Computes the norm of the gradient. Intended to be called every mini-
        batch. Norms of gradients are computed per trainable layer. Also an
        overall norm is computed by concatenating the gradient of all of the
        trainable layers into one vector and computing the norm of that vector.

        Args:
            grad_norms_dict (dict): Keys correspond to the layers (and the overall gradient norm) and
            the values are lists. The lists contain the values of gradients upto the current mini-batch.
            When this function is called, it computes the gradients for each layer and appends the value
            to the corresponding list.
        """
        v = torch.zeros((1,), device=get_device())

        with torch.no_grad():
            for name, param in self.torch_model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        raise ValueError(f"Param {name} has no gradient")
                    layer_grad_norm = torch.linalg.norm(param.grad)

                    if f"grad_{name}_norm" not in grad_norms_dict:
                        grad_norms_dict[f"grad_{name}_norm"] = []

                    grad_norms_dict[f"grad_{name}_norm"].append(layer_grad_norm.item())
                    v = torch.cat((v, param.grad.view(-1)))
            overall_grad_norm = torch.linalg.norm(v)

            if "overall_grad_norm" not in grad_norms_dict:
                grad_norms_dict["overall_grad_norm"] = []

            grad_norms_dict["overall_grad_norm"].append(overall_grad_norm.item())

    def _param_norms(self) -> dict:
        """Computes the norm of the parameters. Intended to be called at the
        end of every epoch. Norms of parameters are computed per  layer. Also
        an overall norm is computed by concatenating all of the layers into one
        vector and computing the norm of that vector.

        Returns:
            A dictionary where keys are string that correspond to the
            layers of the model (and the overall norm) and the values are the norms.
        """
        param_norm_dict = {}
        v = torch.zeros((1,), device=get_device())
        with torch.no_grad():
            for name, param in self.torch_model.named_parameters():
                layer_norm = torch.linalg.norm(param)
                param_norm_dict[f"{name}_norm"] = layer_norm.item()
                v = torch.cat((v, param.view(-1)))
        overall_norm = torch.linalg.norm(v)
        param_norm_dict["params_norm"] = overall_norm.item()
        return param_norm_dict

    @abstractmethod
    def init_loss(self) -> torch.nn.Module:
        """Get the loss function to use when optimizing the model for this
        specific problem.

        Returns:
            The PyTorch loss function,
        """
        pass

    @abstractmethod
    def get_criterions(self) -> List[torch.nn.Module]:
        """Get a list of loss functions to use when evaluating the model for
        this specific problem. Also useful to call when plotting to know what
        evaluation metrics/losses to plot.

        Returns:
            List of PyTorch loss functions.
        """
        pass
