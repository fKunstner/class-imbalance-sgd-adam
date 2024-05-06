from typing import List

import torch

from optexp.problems.problem import Problem
from optexp.problems.fb_problem import FullBatchProblem


class Regression(Problem):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.MSELoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [torch.nn.MSELoss()]


class FullBatchRegression(Regression, FullBatchProblem):
    pass
