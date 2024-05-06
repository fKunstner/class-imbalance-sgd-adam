from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Generator, List

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer


def _check_class_and_same_kwargs(a, b):
    if not isinstance(b, type(a)):
        return False
    if not hasattr(a, "kwargs") or not hasattr(b, "kwargs"):
        return False
    if len(a.kwargs) != len(b.kwargs):
        return False
    for x, y in a.kwargs.items():
        if x not in b.kwargs:
            return False
        if b.kwargs[x] != y:
            return False
    return True


class LayerInit:
    def __init__(self, torch_initializer: Callable, **kwargs) -> None:
        self.torch_initializer = torch_initializer
        self.kwargs = kwargs

    def initialize(self, weight: torch.Tensor) -> None:
        self.torch_initializer(weight, **self.kwargs)

    def __str__(self) -> str:
        init_name = f"torch_initializer={self.torch_initializer.__name__}"
        name = f"Initializer({init_name},"
        for x, y in self.kwargs.items():
            name += f" {x}={y},"
        # remove last comma before adding closing
        name = f"{name[:-1]})"
        return name

    def __iter__(self) -> Generator:
        iters = {"torch_initializer": self.torch_initializer.__name__}
        iters.update(self.kwargs)
        for x, y in iters.items():
            yield x, y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayerInit):
            return False
        return _check_class_and_same_kwargs(self, other)


class Initializer(ABC):
    def __init__(self, **kwargs: LayerInit) -> None:
        self.kwargs = kwargs
        self.param_list: List[str] = []

    def check_args(self):
        for name, _ in self.kwargs.items():
            if name not in self.param_list:
                raise ValueError(f"{name} is not a layer in the model")

    def __repr__(self) -> str:
        name = f"{self.__class__.__name__}("
        before = len(name)
        for x, y in self.kwargs.items():
            name += f" {x}={y},"

        # if there were no kwargs then just add a closing bracket
        if len(name) == before:
            return f"{name})"

        # remove last comma before adding closing bracket otherwise
        return f"{name[:-1]})"

    def __iter__(self) -> Generator:
        for x, y in self.kwargs.items():
            yield x, y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Initializer):
            return False
        return _check_class_and_same_kwargs(self, other)

    @abstractmethod
    def initialize(self, model):
        raise NotImplementedError


class EncoderDecoderInitializer(Initializer):
    def __init__(self, **kwargs: LayerInit) -> None:
        super(EncoderDecoderInitializer, self).__init__(**kwargs)
        self.param_list = []
        self.check_args()

    def initialize(self, model):
        raise NotImplementedError


class VisionTransformerInitializer(Initializer):
    def __init__(self, **kwargs: LayerInit) -> None:
        super(VisionTransformerInitializer, self).__init__(**kwargs)
        self.param_list = [
            "ln_1_weight",
            "ln_1_bias",
            "in_proj_weight",
            "in_proj_bias",
            "out_proj_weight",
            "out_proj_bias",
            "ln_2_weight",
            "ln_2_bias",
            "ln_weight",
            "ln_bias",
            "heads_weight",
            "heads_bias",
        ]
        self.check_args()

    def initialize(self, model):
        raise NotImplementedError


class TransformerEncoderInitializer(Initializer):
    def __init__(self, **kwargs: LayerInit):
        super(TransformerEncoderInitializer, self).__init__(**kwargs)
        self.param_list = [
            "in_proj_weight",
            "in_proj_bias",
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "out_proj_weight",
            "out_proj_bias",
            "linear1_weight",
            "linear1_bias",
            "linear2_weight",
            "linear2_bias",
            "norm1_weight",
            "norm1_bias",
            "norm2_weight",
            "norm2_bias",
            "embedding_weight",
            "linear_weight",
            "linear_bias",
        ]
        self.check_args()

    def initialize(self, model):
        for layer in model.transformer_encoder.layers:
            self.set_multihead_attention(layer)
        self.set_embedding_and_linear(model)

    def init(self, param_name, layer, *path_to_args, default_init=None):
        if layer is None:
            return

        def get_param():
            param = layer
            for path in path_to_args:
                if param is not None:
                    param = getattr(param, path, None)
            return param

        param = get_param()

        if param is None:
            return

        if param_name in self.kwargs:
            self.kwargs[param_name].initialize(param)
        elif default_init is not None:
            default_init(param)

    def set_multihead_attention(self, encoder_layer: TransformerEncoderLayer):
        if encoder_layer.self_attn._qkv_same_embed_dim:
            self.init("in_proj_weight", encoder_layer, "self_attn", "in_proj_weight")
        else:
            self.init("q_proj_weight", encoder_layer, "self_attn", "q_proj_weight")
            self.init("k_proj_weight", encoder_layer, "self_attn", "k_proj_weight")
            self.init("v_proj_weight", encoder_layer, "self_attn", "v_proj_weight")

        self.init("in_proj_bias", encoder_layer, "self_attn", "in_proj_bias")
        self.init("bias_k", encoder_layer, "self_attn", "bias_k")
        self.init("bias_v", encoder_layer, "self_attn", "bias_v")
        self.init("linear1_weight", encoder_layer, "linear1", "weight")
        self.init("linear1_bias", encoder_layer, "linear1", "bias")
        self.init("linear2_weight", encoder_layer, "linear2", "weight")
        self.init("linear2_bias", encoder_layer, "linear2", "bias")
        self.init("norm1_weight", encoder_layer, "norm1", "weight")
        self.init("norm1_bias", encoder_layer, "norm1", "bias")
        self.init("norm2_weight", encoder_layer, "norm2", "weight")
        self.init("norm2_bias", encoder_layer, "norm2", "bias")

    def set_embedding_and_linear(self, model):
        self.init(
            "linear_bias", model.classifier_head, "bias", default_init=nn.init.zeros_
        )
        self.init(
            "linear_weight",
            model.classifier_head,
            "weight",
            default_init=partial(nn.init.uniform_, a=-0.1, b=0.1),
        )
        self.init(
            "embedding_weight",
            model.embedding,
            "weight",
            default_init=partial(nn.init.uniform_, a=-0.1, b=0.1),
        )

    @staticmethod
    def default(
        init_std=0.02,
    ) -> "TransformerEncoderInitializer":
        weight_init = LayerInit(nn.init.normal_, mean=0.0, std=init_std)
        bias_init = LayerInit(nn.init.constant_, val=0.0)
        layer_norm_weight = LayerInit(nn.init.normal_, mean=1.0, std=init_std)
        return TransformerEncoderInitializer(
            linear1_weight=weight_init,
            linear1_bias=bias_init,
            linear2_weight=weight_init,
            linear2_bias=bias_init,
            linear_weight=weight_init,
            linear_bias=bias_init,
            embedding_weight=weight_init,
            norm1_weight=layer_norm_weight,
            norm1_bias=bias_init,
            norm2_weight=layer_norm_weight,
            norm2_bias=bias_init,
        )

    @staticmethod
    def default_short(
        init_std=0.02,
    ) -> "TransformerEncoderInitializer":
        weight_init = LayerInit(nn.init.normal_, mean=0.0, std=init_std)
        bias_init = LayerInit(nn.init.constant_, val=0.0)
        return TransformerEncoderInitializer(
            linear_weight=weight_init,
            linear_bias=bias_init,
            embedding_weight=weight_init,
        )

    @staticmethod
    def default_scaled_by_depth(depth: int, init_std: float = 0.02):
        INIT_STD = init_std

        weight_init = LayerInit(nn.init.normal_, mean=0.0, std=INIT_STD)
        residual_weight_init = LayerInit(
            nn.init.normal_, mean=0.0, std=INIT_STD / (2 * depth) ** 0.5
        )
        bias_init = LayerInit(nn.init.constant_, val=0.0)
        layer_norm_weight = LayerInit(nn.init.constant_, val=1.0)

        return TransformerEncoderInitializer(
            linear1_weight=weight_init,
            linear1_bias=bias_init,
            linear2_weight=residual_weight_init,
            linear2_bias=bias_init,
            out_proj_weight=residual_weight_init,
            linear_weight=weight_init,
            linear_bias=bias_init,
            embedding_weight=weight_init,
            norm1_weight=layer_norm_weight,
            norm1_bias=bias_init,
            norm2_weight=layer_norm_weight,
            norm2_bias=bias_init,
        )
