from dataclasses import dataclass

import torch

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer


@dataclass
class AdamW(Optimizer):
    """Wrapper class for defining and loading the Adam optimizer."""

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    split_decay: bool = False

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.split_decay:
            # based on karpathy
            decay = set()
            no_decay = set()
            named_parameters = model.named_parameters()
            named_parameter_count = 0
            for pn, p in named_parameters:
                named_parameter_count += 1
                if pn.endswith("bias"):
                    no_decay.add(p)
                elif "norm" in pn:
                    no_decay.add(p)
                elif "embedding" in pn:
                    no_decay.add(p)
                else:
                    decay.add(p)
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0
            assert len(union_params) == named_parameter_count
            param_groups = [
                {"params": list(decay), "weight_decay": self.weight_decay},
                {"params": list(no_decay), "weight_decay": 0.0},
            ]
            return torch.optim.AdamW(
                param_groups,
                lr=self.learning_rate.as_float(),
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate.as_float(),
                betas=(self.beta1, self.beta2),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )


def AdamW_NM(lr: LearningRate) -> AdamW:
    return AdamW(learning_rate=lr, beta1=0)


def AdamW_M(lr: LearningRate) -> AdamW:
    return AdamW(learning_rate=lr, beta1=0.9)
