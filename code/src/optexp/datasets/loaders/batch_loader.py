import torch


class BatchLoader:
    def __init__(
        self, data: torch.Tensor, targets: torch.Tensor, batch_size: int
    ) -> None:
        self.data = data
        self.targets = targets
        assert self.data.shape[0] == self.targets.shape[0]
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i in range(0, self.data.shape[0]):
            data, targets = self.get_batch(self.i)
            self.i += self.batch_size
            return data, targets
        else:
            raise StopIteration

    def get_batch(self, i: int):
        if i + self.batch_size >= self.data.shape[0]:
            num_samples = self.data.shape[0] - i
        else:
            num_samples = self.batch_size
        data = self.data[i : i + num_samples]
        targets = self.targets[i : i + num_samples]
        return data, targets
