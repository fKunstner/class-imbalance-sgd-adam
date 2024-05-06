import copy
from typing import Optional
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch.optim.optimizer import Optimizer

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer as Opt


@dataclass
class LineSearchSGD(Opt):
    gamma: float = 0.5
    beta: float = 0.9

    def load(self, model) -> Optimizer:
        return GDLineSearch(
            model.parameters(),
            max_lr=self.learning_rate.as_float(),
            gamma=self.gamma,
            beta=self.beta,
        )


def LSGD() -> LineSearchSGD:
    return LineSearchSGD(learning_rate=LearningRate(Fraction(0, 1)))


class GDLineSearch(Optimizer):
    def __init__(
        self,
        params,
        max_lr,
        gamma,
        beta,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize=False,
        foreach: Optional[bool] = None,
        differentiable=False
    ):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            max_lr=max_lr,
            gamma=gamma,
            beta=beta,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GDLineSearch, self).__init__(params, defaults)
        self.state["step_size"] = max_lr

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grad_norm = 0
            params_current = copy.deepcopy(group["params"])

            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    grad_norm += torch.sum(torch.mul(p.grad, p.grad))
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            grad_norm = torch.sqrt(grad_norm)

            if grad_norm >= 1e-8:
                step_size = 1.5 * self.state["step_size"]
                while True:
                    try_sgd_update(group["params"], step_size, params_current, d_p_list)
                    loss_next = closure()
                    found, step_size = check_armijo_condition(
                        step_size,
                        loss,
                        loss_next,
                        grad_norm,
                        group["gamma"],
                        group["beta"],
                    )
                    if found:
                        self.state["step_size"] = step_size
                        break

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def try_sgd_update(params, step_size, params_current, d_p_list):
    for i, param in enumerate(params):
        if d_p_list[i] is not None:
            param.data = params_current[i] - step_size * d_p_list[i]


def check_armijo_condition(step_size, loss, loss_next, grad_norm, gamma, beta):
    found = False
    break_condition = loss_next - (loss - (step_size) * gamma * grad_norm**2)

    if break_condition <= 0:
        found = True
    else:
        step_size = step_size * beta

    return found, step_size
