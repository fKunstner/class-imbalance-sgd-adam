from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector as p2v
from torch.optim.optimizer import Optimizer

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer as Opt


@dataclass
class Sign(Opt):
    """Wrapper class for defining and loading the Sign optimizer."""

    momentum: float = 0

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return SignSGD(
            model.parameters(), lr=self.learning_rate.as_float(), momentum=self.momentum
        )


def Sign_M(lr: LearningRate) -> Sign:
    return Sign(learning_rate=lr, momentum=0.9)


def Sign_NM(lr: LearningRate) -> Sign:
    return Sign(learning_rate=lr, momentum=0)


@dataclass
class NormSGD(Opt):
    """Wrapper class for defining and loading the Normalized SGD optimizer."""

    momentum: float = 0

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return NormalizedSGD(
            model.parameters(), lr=self.learning_rate.as_float(), momentum=self.momentum
        )


def NormSGD_M(lr: LearningRate) -> NormSGD:
    return NormSGD(learning_rate=lr, momentum=0.9)


def NormSGD_NM(lr: LearningRate) -> NormSGD:
    return NormSGD(learning_rate=lr, momentum=0)


@dataclass
class ScaledSign(Opt):
    """Wrapper class for defining and loading ReNormalized Sign Descent."""

    momentum: float = 0

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return RescaledSignDescent(
            model.parameters(), lr=self.learning_rate.as_float(), momentum=self.momentum
        )


def ScaledSign_M(lr: LearningRate) -> ScaledSign:
    return ScaledSign(learning_rate=lr, momentum=0.9)


def ScaledSign_NM(lr: LearningRate) -> ScaledSign:
    return ScaledSign(learning_rate=lr, momentum=0)


class CopyOfSGD(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CopyOfSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def _step_with_direction(self, closure=None, direction_func=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # grad norm comp
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    # norm decent
                    d_p_list.append(direction_func(p.grad))

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss

    @staticmethod
    def _eval_closure(closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def _total_grad_norm(self, norm: float):
        total_grad = p2v(
            [
                p.grad if p.grad is not None else torch.zeros_like(p)
                for group in self.param_groups
                for p in group["params"]
            ]
        )
        return total_grad.norm(norm)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self._eval_closure(closure)
        self._step_with_direction(closure, lambda g: g)
        return loss


def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


class PlainSGD(CopyOfSGD):
    def __init__(self, params, lr, momentum: float = 0):
        super(PlainSGD, self).__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=0,
            nesterov=False,
        )


class BlockNormalizedSGD(PlainSGD):
    """Change the magnitude and direction, but by block rather than
    coordinate."""

    def __init__(self, params, lr, momentum: float = 0, norm: float = 1):
        assert norm > 0
        self.norm = norm
        super(BlockNormalizedSGD, self).__init__(params, momentum=momentum, lr=lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._eval_closure(closure)
        self._step_with_direction(closure, lambda g: g / g.norm(self.norm))
        return loss


class RescaledSignDescent(PlainSGD):
    """Change the direction using the sign but keep the magnitude."""

    def __init__(self, params, lr, momentum: float = 0, norm: float = 1):
        assert norm > 0
        self.norm = norm
        super(RescaledSignDescent, self).__init__(params, lr=lr, momentum=momentum)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._eval_closure(closure)
        total_grad_norm = self._total_grad_norm(self.norm)
        self._step_with_direction(closure, lambda g: torch.sign(g) * total_grad_norm)
        return loss


class NormalizedSGD(PlainSGD):
    """Change the magnitude but keep the direction."""

    def __init__(self, params, lr, momentum=0, norm: float = 2):
        assert norm > 0
        self.norm = norm
        super(NormalizedSGD, self).__init__(params, lr=lr, momentum=momentum)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._eval_closure(closure)
        total_grad_norm = self._total_grad_norm(self.norm)
        self._step_with_direction(closure, lambda g: g / total_grad_norm)
        return loss


class SignSGD(PlainSGD):
    """Change the magnitude and direction."""

    def __init__(self, params, lr, momentum=0):
        super(SignSGD, self).__init__(params, lr=lr, momentum=momentum)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._eval_closure(closure)
        self._step_with_direction(closure, lambda g: torch.sign(g))
        return loss
