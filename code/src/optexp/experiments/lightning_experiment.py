import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as ptl
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from optexp import Experiment, config
from optexp.config import get_device, get_logger
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.optimizers import Optimizer
from optexp.problems import DivergingException, Problem

# from optexp.experiments import Experiment


# Note, in some cases, it won't kill all the gpu processes. before  figuring out how to   actually fix this, do nvidia-smi, see the PIDs listed, and do
# kill -9 <PID> for each of them. attempting to figure out an automated shutdown system
@dataclass
class LightningExperiment(Experiment):

    steps: int = 5000
    nodes: int = 1
    devices: int = -1
    strategy: str = "ddp"
    eval_every: int = 50
    gradient_acc_steps: int = 16
    wandb_autosync: bool = True

    def __post_init__(self):
        if self.epochs is not None and self.epochs != 0:
            raise ValueError(
                "LightningExperiment setup with epochs, but onlysupports steps"
            )

    def run_experiment(self) -> None:
        """
        Performs a run of the experiment. Generates the run-id, applies the seed
        and creates the data logger. Initializes the problem and optimizer and
        optimizes the problem given the optimizer for the defined amount of epochs.
        Logs the loss function values/metrics returned during the eval and training.
        Catches any exception raised during this process and logs it before exiting.

        Raises:
            BaseException: Raised when user Ctrl+C when experiment is running.
        """
        run_id = time.strftime("%Y-%m-%d--%H-%M-%S")

        self._apply_seed()

        self.fabric = ptl.Fabric(
            accelerator=get_device(),
            devices=self.devices,
            num_nodes=self.nodes,
            strategy=self.strategy,
        )
        self.fabric.launch()

        # only rank 0 gets a real one
        data_logger = None
        self.fabric.barrier()
        if self.fabric.global_rank == 0:
            data_logger = DataLogger(
                config_dict=asdict_with_class(self),
                group=self.group,
                run_id=run_id,
                exp_id=self.exp_id(),
                save_directory=self.save_directory(),
                wandb_autosync=self.wandb_autosync,
            )

            get_logger().info("=" * 80)
            get_logger().info(f"Initializing  experiment: {self}")
            get_logger().info("=" * 80)
        self.fabric.barrier()
        exceptions = {
            "DivergenceException": False,
            "Exception": False,
            "BaseException": False,
        }
        experiment_success = True

        try:
            with self.fabric.rank_zero_first():
                self.problem.init_problem()
            
            opt = self.optim.load(self.problem.torch_model)

            self.problem.torch_model, opt = self.fabric.setup(
                self.problem.torch_model, opt
            )
            self.problem.train_loader, self.problem.val_loader = (
                self.fabric.setup_dataloaders(
                    self.problem.train_loader,
                    self.problem.val_loader,
                    move_to_device=True,
                )
            )

            if self.problem.is_mixed_batch:
                self.problem.eval_train_loader, self.problem.eval_val_loader = (
                    self.fabric.setup_dataloaders(
                        self.problem.eval_train_loader,
                        self.problem.eval_val_loader,
                        move_to_device=True,
                    )
                )
            metrics_and_counts_eval_train = self.problem.eval(
                val=False, return_raw=True
            )
            metrics_and_counts_eval_val = self.problem.eval(val=True, return_raw=True)

            metrics_eval_train = self.aggregate_metrics(
                metrics_and_counts_eval_train, key_prefix="tr"
            )
            metrics_eval_val = self.aggregate_metrics(
                metrics_and_counts_eval_val, key_prefix="val"
            )

            self.synchronised_log(
                data_logger, metrics_eval_val, metrics_eval_train, {"step": 0}
            )

            torch.cuda.empty_cache()

            train_loss = 0.0
            num_samples = 0
            mini_batch_losses = []
            grad_norms: Dict[str, List] = {}
            loader = iter(self.problem.train_loader)

            def get_batch():
                nonlocal loader
                try:
                    features, labels = next(loader)
                except StopIteration:
                    loader = iter(self.problem.train_loader)
                    features, labels = next(loader)
                return features, labels

            opt.zero_grad()

            for t in range(1, self.steps + 1):
                loss_to_save: float = 0.0

                for t_acc in range(self.gradient_acc_steps):
                    features, labels = get_batch()
                    y_pred = self.problem.torch_model(features)
                    loss = (
                        self.problem.criterion(y_pred, labels) / self.gradient_acc_steps
                    )
                    loss_to_save += loss.item()

                    if math.isnan(loss) or math.isinf(loss):
                        exceptions["DivergenceException"] = True

                    train_loss += loss.item() * len(features) * self.gradient_acc_steps
                    num_samples += len(features)

                    self.fabric.backward(loss)
                    del y_pred
                    del features
                    del labels

                self._grad_norms(grad_norms_dict=grad_norms)
                mini_batch_loss = loss_to_save * self.gradient_acc_steps
                reduced_mini_batch_loss: Tensor = self.fabric.all_reduce(
                    mini_batch_loss, reduce_op="mean"
                )
                mini_batch_losses.append(reduced_mini_batch_loss.cpu().item())
                del reduced_mini_batch_loss

                opt.step()
                opt.zero_grad()
                torch.cuda.empty_cache()

                if t % self.eval_every == 0:

                    metrics_and_counts_eval_train = self.problem.eval(
                        val=False, return_raw=True
                    )
                    metrics_and_counts_eval_val = self.problem.eval(
                        val=True, return_raw=True
                    )
                    live_train_loss = train_loss / num_samples
                    reduced_live_train_loss = self.fabric.all_reduce(
                        live_train_loss, reduce_op="mean"
                    )
                    metrics_training = {
                        "live_train_loss": reduced_live_train_loss.cpu().item(),
                        "mini_batch_losses": mini_batch_losses,
                    }
                    del reduced_live_train_loss
                    metrics_eval_train = self.aggregate_metrics(
                        metrics_and_counts_eval_train, key_prefix="tr"
                    )
                    metrics_eval_val = self.aggregate_metrics(
                        metrics_and_counts_eval_val, key_prefix="val"
                    )

                    param_norms = self._param_norms()
                    metrics_training.update(param_norms)
                    metrics_training.update(grad_norms)

                    self.synchronised_log(
                        data_logger,
                        metrics_eval_val,
                        metrics_eval_train,
                        {"step": t},
                        metrics_training,
                    )
                    torch.cuda.empty_cache()

                    train_loss = 0.0
                    num_samples = 0
                    mini_batch_losses = []
                    grad_norms = {}

                self.check_exceptions(exceptions)

        except DivergingException as e:
            exceptions["DivergenceException"] = True
            experiment_success = False
            if self.fabric.global_rank == 0:
                get_logger().warning("TERMINATING EARLY. Diverging.")
                get_logger().warning(e, exc_info=True)
                if data_logger is not None:
                    data_logger.save(exit_code=0)
        except Exception as e:
            exceptions["Exception"] = True
            experiment_success = False
            if self.fabric.global_rank == 0:
                get_logger().error("TERMINATING. Encountered error")
                get_logger().error(e, exc_info=True)
                if data_logger is not None:
                    data_logger.save(exit_code=1)
        except BaseException as e:
            exceptions["BaseException"] = True
            experiment_success = False
            if self.fabric.global_rank == 0:
                get_logger().error("TERMINATING. System exit")
                get_logger().error(e, exc_info=True)
                if data_logger is not None:
                    data_logger.save(exit_code=1)
        finally:
            self.fabric.barrier()
            if self.fabric.global_rank == 0:
                get_logger().info("All Processes Terminating")

        experiment_success = any(self.fabric.all_gather(experiment_success))
        if self.fabric.global_rank == 0 and experiment_success:
            get_logger().info("Experiment finished.")
            if data_logger is not None:
                data_logger.save(exit_code=0)

    def check_exceptions(self, exceptions: Dict[str, bool]) -> None:
        all_exceptions = self.fabric.all_gather(exceptions)
        if any(all_exceptions["DivergenceException"]):
            raise DivergingException()
        elif any(all_exceptions["Exception"]):
            raise Exception()
        elif any(all_exceptions["BaseException"]):
            raise BaseException

    def aggregate_metrics(self, metrics_and_counts: Dict, key_prefix: str) -> Dict:
        self.fabric.barrier()
        metric_values = metrics_and_counts["metric_values"]
        metric_n_samples = metrics_and_counts["metric_n_samples"]
        total_samples = metrics_and_counts["total_samples"]
        total_samples = self.fabric.all_reduce(total_samples, reduce_op="sum").item()

        for module in metric_values.keys():
            metric_values[module] = self.fabric.all_reduce(
                metric_values[module], reduce_op="sum"
            )
            if module in metric_n_samples.keys():
                metric_n_samples[module] = self.fabric.all_reduce(
                    metric_n_samples[module], reduce_op="sum"
                )
                metric_values[module] = metric_values[module] / metric_n_samples[module]
            else:
                metric_values[module] = metric_values[module] / total_samples

        metrics = {
            f"{key_prefix}_{str(module)[:-2]}": (
                value.cpu().tolist() if torch.numel(value) > 1 else value.cpu().item()
            )
            for module, value in metric_values.items()
        }
        del metric_values
        return metrics

    def synchronised_log(
        self,
        data_logger: DataLogger | None,
        metrics_eval_val: Dict[str, Any],
        metrics_eval_train: Dict[str, Any],
        step_dict: Dict[str, int],
        metrics_training: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fabric.barrier()
        if self.fabric.global_rank == 0 and data_logger is not None:
            if metrics_training is not None:
                data_logger.log_data(metrics_training)
            data_logger.log_data(metrics_eval_val)
            data_logger.log_data(metrics_eval_train)
            data_logger.log_data(step_dict)
            data_logger.commit()
        self.fabric.barrier()

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
            for name, param in self.problem.torch_model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        raise ValueError(f"Param {name} has no gradient")
                    layer_grad_norm = torch.linalg.norm(param.grad)

                    if f"grad_{name}_norm" not in grad_norms_dict:
                        grad_norms_dict[f"grad_{name}_norm"] = []

                    grad_norms_dict[f"grad_{name}_norm"].append(
                        layer_grad_norm.cpu().item()
                    )
                    v = torch.cat((v, param.grad.view(-1)))
                    del layer_grad_norm
            overall_grad_norm = torch.linalg.norm(v)

            if "overall_grad_norm" not in grad_norms_dict:
                grad_norms_dict["overall_grad_norm"] = []

            grad_norms_dict["overall_grad_norm"].append(overall_grad_norm.cpu().item())
            del overall_grad_norm

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
            for name, param in self.problem.torch_model.named_parameters():
                layer_norm = torch.linalg.norm(param)
                param_norm_dict[f"{name}_norm"] = layer_norm.cpu().item()
                del layer_norm
                v = torch.cat((v, param.view(-1)))
        overall_norm = torch.linalg.norm(v)
        param_norm_dict["params_norm"] = overall_norm.cpu().item()
        del overall_norm
        return param_norm_dict
