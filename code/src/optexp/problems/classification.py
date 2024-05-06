import warnings
from dataclasses import dataclass
from typing import List

import torch

from optexp.problems.fb_problem import FullBatchProblem
from optexp.problems.problem import Problem
from optexp.problems.utils import (
    Accuracy,
    AccuracyPerClass,
    ClassificationSquaredLoss,
    ClassificationWeightedSquaredLoss,
    CrossEntropyLossPerClass,
    LogitAdjustedCrossEntropyLoss,
    MSELossPerClass,
    WeightedCrossEntropyLoss,
    WeightedCrossEntropyLossPerClass,
)


class Classification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [torch.nn.CrossEntropyLoss(), Accuracy()]


class ClassificationWithPerClassStats(Classification):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
            CrossEntropyLossPerClass(),
            AccuracyPerClass(),
        ]


class SquaredLossWithPerClassStats(Problem):
    def init_loss(self) -> torch.nn.Module:
        return ClassificationSquaredLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            ClassificationSquaredLoss(),
            Accuracy(),
            MSELossPerClass(),
            AccuracyPerClass(),
        ]


class LogitAdjustedClassification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return LogitAdjustedCrossEntropyLoss(class_porps=self.compute_porps())

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            LogitAdjustedCrossEntropyLoss(class_porps=self.compute_porps()),
            Accuracy(),
            torch.nn.CrossEntropyLoss(),
        ]

    def compute_porps(self):
        if not hasattr(self, "class_freqs"):
            warnings.warn(
                "Trying to compute weights before class_freqs are set, returning uniform weights."
            )
            return torch.ones(1)

        return self.class_freqs / torch.sum(self.class_freqs)


@dataclass
class WeightedClassification(Problem):
    sqrt_weights: bool = False

    def init_loss(self) -> torch.nn.Module:
        return WeightedCrossEntropyLoss(weights=self.compute_weights())

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            WeightedCrossEntropyLoss(weights=self.compute_weights()),
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
        ]

    def compute_weights(self):
        if not hasattr(self, "class_freqs"):
            warnings.warn(
                "Trying to compute weights before class_freqs are set, returning uniform weights."
            )
            return torch.ones(1)

        if self.sqrt_weights:
            return 1.0 / torch.sqrt(self.class_freqs)
        else:
            return 1.0 / self.class_freqs


@dataclass
class SqauredWeightedClassification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return WeightedCrossEntropyLoss(weights=self.compute_weights())

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            WeightedCrossEntropyLoss(weights=self.compute_weights()),
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
        ]

    def compute_weights(self):
        if not hasattr(self, "class_freqs"):
            warnings.warn(
                "Trying to compute weights before class_freqs are set, returning uniform weights."
            )
            return torch.ones(1)

        return 1.0 / torch.square(self.class_freqs)


@dataclass
class WeightedClassificationWithPerClassStats(WeightedClassification):
    def init_loss(self) -> torch.nn.Module:
        return WeightedCrossEntropyLoss(weights=self.compute_weights())

    def get_criterions(self) -> List[torch.nn.Module]:
        x = self.compute_weights()
        return [
            WeightedCrossEntropyLoss(weights=x),
            WeightedCrossEntropyLossPerClass(weights=x),
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
            AccuracyPerClass(),
            CrossEntropyLossPerClass(),
        ]


@dataclass
class WeightedSquaredLossClassification(Problem):
    sqrt_weights: bool = False

    def init_loss(self) -> torch.nn.Module:
        return ClassificationWeightedSquaredLoss(weights=self.compute_weights())

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            ClassificationWeightedSquaredLoss(weights=self.compute_weights()),
            ClassificationSquaredLoss(),
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
        ]

    def compute_weights(self):
        if not hasattr(self, "class_freqs"):
            warnings.warn(
                "Trying to compute weights before class_freqs are set, returning uniform weights."
            )
            return torch.ones(1)

        if self.sqrt_weights:
            return 1.0 / torch.sqrt(self.class_freqs)
        else:
            return 1.0 / self.class_freqs


class SquaredLossClassification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return ClassificationSquaredLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [ClassificationSquaredLoss(), Accuracy()]


class FullBatchClassification(Classification, FullBatchProblem):
    pass


class FullBatchClassificationWithPerClassStats(
    ClassificationWithPerClassStats, FullBatchProblem
):
    pass


class FullBatchSquaredLossClassificationWithPerClassStats(
    SquaredLossWithPerClassStats, FullBatchProblem
):
    pass


class FullBatchSquaredLossClassification(SquaredLossClassification, FullBatchProblem):
    pass


class FullBatchWeightedClassification(WeightedClassification, FullBatchProblem):
    pass


class FullBatchSqauredWeightedClassification(
    SqauredWeightedClassification, FullBatchProblem
):
    pass


class FullBatchWeightedClassificationWithPerClassStats(
    WeightedClassificationWithPerClassStats, FullBatchProblem
):
    pass


class FullBatchLogitAdjustedClassification(
    LogitAdjustedClassification, FullBatchProblem
):
    pass


class FullBatchWeightedSquaredLossClassification(
    WeightedSquaredLossClassification, FullBatchProblem
):
    pass
