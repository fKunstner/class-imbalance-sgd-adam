from dataclasses import dataclass

from torch import nn

from optexp.config import get_logger
from optexp.models.model import Model


@dataclass
class LinearInit0(Model):
    input: int
    output: int
    bias: bool = False

    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        model = nn.Linear(self.input, self.output, bias=self.bias)
        model.weight.data *= 0
        if self.bias:
            model.bias.data *= 0
        return model
