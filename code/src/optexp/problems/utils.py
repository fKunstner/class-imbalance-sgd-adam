import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, mse_loss


class Accuracy(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, labels):
        classes = torch.argmax(inputs, dim=1)
        if self.reduction == "mean":
            return torch.mean((classes == labels).float())
        else:
            return torch.sum((classes == labels).float())


def _groupby_average(inputs, classes, num_classes):
    """Given an [n] inputs tensor and an [n] classes tensor containing class
    indices in [1, ..., C], returns a [C] tensor containing the average ```
    out[c] = sum(inputs[classes == c]) ```"""
    classes = classes.view(-1)

    label_counts = torch.zeros(num_classes, dtype=torch.float, device=classes.device)
    label_counts = label_counts.scatter_add_(0, classes, torch.ones_like(inputs))

    sum_by_class = torch.zeros(num_classes, dtype=torch.float, device=classes.device)
    sum_by_class = sum_by_class.scatter_add_(0, classes, inputs)

    return sum_by_class, label_counts


class CrossEntropyLossPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, reduction="none")
        return _groupby_average(losses, labels, num_classes)


class MSELossPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        losses = mse_loss(inputs, one_hot_labels, reduction="none")
        return _groupby_average(losses.mean(axis=1), labels, num_classes)


class AccuracyPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        classes = torch.argmax(inputs, dim=1)
        accuracy_per_sample = (classes == labels).float()
        return _groupby_average(accuracy_per_sample, labels, num_classes)


class ClassificationSquaredLoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        class_sum = torch.sum(
            (torch.masked_select(inputs, one_hot_labels > 0) - 1) ** 2
        )
        output = (1.0 / num_classes) * (
            class_sum
            + torch.sum(torch.square(torch.masked_select(inputs, one_hot_labels == 0)))
        )
        return output


class ClassificationWeightedSquaredLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, labels):
        num_classes = inputs.shape[1]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)

        multiplier = self.weights.gather(0, labels)

        true_classes = torch.sum(
            multiplier * ((torch.masked_select(inputs, one_hot_labels > 0) - 1) ** 2)
        )

        false_classes = torch.sum(
            multiplier
            * torch.sum(
                torch.square(torch.masked_select(inputs, one_hot_labels == 0)).reshape(
                    -1, num_classes - 1
                ),
                dim=1,
            ),
        )

        return (1.0 / num_classes) * (true_classes + false_classes)


def reshape_for_per_sequence(batch_size, seq_len, inputs, labels):
    inputs = inputs.reshape(
        (inputs.shape[0] // batch_size, batch_size, inputs.shape[1])
    )
    labels = labels.reshape((labels.shape[0] // batch_size, batch_size))
    inputs_tgt_i = inputs[seq_len]
    labels_tgt_i = labels[seq_len]
    return inputs_tgt_i, labels_tgt_i


class LogitAdjustedCrossEntropyLoss(nn.Module):
    def __init__(self, class_porps: torch.Tensor, tau=1.0):
        super().__init__()
        self.props = class_porps.unsqueeze(0)
        self.tau = tau

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        return cross_entropy(inputs + self.tau * torch.log(self.props), labels)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        loss = cross_entropy(inputs, labels, weight=self.weights, reduction="sum")
        weights = torch.sum(self.weights[labels])
        return loss, weights


class WeightedCrossEntropyLossPerClass(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, weight=self.weights, reduction="none")
        losses, counts = _groupby_average(losses, labels, num_classes)
        weights = counts * self.weights
        return losses, weights


class AccuracyPerSequenceLength(nn.Module):
    def __init__(self, seq_len: int, batch_size: int, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.batch_size = batch_size
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        if inputs.shape[0] // self.batch_size <= self.seq_len:
            return torch.tensor(0.0)

        inputs_tgt_i, labels_tgt_i = reshape_for_per_sequence(
            self.batch_size, self.seq_len, inputs, labels
        )

        classes_i = torch.argmax(inputs_tgt_i, dim=1)

        if self.reduction == "mean":
            return torch.mean((classes_i == labels_tgt_i).float())
        else:
            return torch.sum((classes_i == labels_tgt_i).float())

    def __str__(self) -> str:
        x = f"{super().__str__()[:-2]}_{self.seq_len}()"
        return x


class CrossEntropyPerSequenceLength(nn.Module):
    def __init__(self, seq_len: int, batch_size: int, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.batch_size = batch_size
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        if inputs.shape[0] // self.batch_size <= self.seq_len:
            return torch.tensor(0.0)

        inputs_tgt_i, labels_tgt_i = reshape_for_per_sequence(
            self.batch_size, self.seq_len, inputs, labels
        )

        x = torch.nn.CrossEntropyLoss()(inputs_tgt_i, labels_tgt_i)
        return x

    def __str__(self) -> str:
        x = f"{super().__str__()[:-2]}_{self.seq_len}()"
        return x


class DivergingException(Exception):
    """Called when loss is NAN or INF."""

    def __init__(self, message="Live training loss is NAN or INF") -> None:
        self.message = message
        super().__init__(self.message)
