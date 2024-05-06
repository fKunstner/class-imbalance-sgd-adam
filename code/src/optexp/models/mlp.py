from dataclasses import dataclass
from typing import List, Optional, Type

import torch.nn
from torch import nn

from optexp.config import get_logger
from optexp.models.model import Model


@dataclass
class MLP(Model):
    hidden_layers: Optional[List[int]]
    activation: Optional[Type[torch.nn.Module]]

    def load_model(self, input_shape, output_shape):
        get_logger().info("Creating model: " + self.__class__.__name__)
        x = MLPTorch(input_shape, output_shape, self.hidden_layers, self.activation)
        return x


class MLPTorch(nn.Module):
    def __init__(
        self, input_shape, output_shape, hidden_layer_list, activation
    ) -> None:
        super(MLPTorch, self).__init__()
        self.model = self.create_layers(
            input_shape, output_shape, hidden_layer_list, activation
        )

    @staticmethod
    def create_layers(input_shape, output_shape, layer_list, activation):
        mlp = nn.Sequential()
        if layer_list is None:
            return nn.Linear(input_shape[0], output_shape[0])

        mlp.add_module("input_layer", nn.Linear(input_shape[0], layer_list[0]))
        if activation:
            mlp.add_module("act_0", activation())

        for i, layer in enumerate(layer_list):
            if i == len(layer_list) - 1:
                mlp.add_module(
                    f"hidden_layer_{i+1}",
                    nn.Linear(layer, output_shape[0]),
                )
            else:
                mlp.add_module(
                    f"hidden_layer_{i+1}",
                    nn.Linear(layer, layer_list[i + 1]),
                )
                if activation:
                    mlp.add_module(f"act_{i+1}", activation())
        return mlp

    def forward(self, x):
        return self.model(x)
