import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import functional as F

from optexp.models.initializer import TransformerEncoderInitializer
from optexp.models.model import Model


@dataclass
class TransformerModel(Model):
    num_heads: int
    depth: int
    width_mlp: int
    emb_dim: int
    drop_out: float
    init: TransformerEncoderInitializer

    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> nn.Module:
        model = _TransformerTorch(
            ntoken=input_shape[0],
            d_model=self.emb_dim,
            nhead=self.num_heads,
            d_hid=self.width_mlp,
            nlayers=self.depth,
            dropout=self.drop_out,
        )
        self.init.initialize(model)
        return model


@dataclass
class GPTModel(Model):
    num_heads: int
    depth: int
    emb_dim: int
    init: TransformerEncoderInitializer
    width_mlp: int | None = None

    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> nn.Module:
        model = _GPTModifiable(
            ntoken=input_shape[0],
            d_model=self.emb_dim,
            nhead=self.num_heads,
            d_hid=self.width_mlp,
            nlayers=self.depth,
        )
        self.init.initialize(model)
        return model


@dataclass
class BasicTransformerModel(Model):
    num_heads: int
    depth: int
    width_mlp: int | None
    emb_dim: int
    drop_out: float
    init: TransformerEncoderInitializer
    layer_norm: bool = True
    pos_encoding: bool = True

    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> nn.Module:
        need_linear = self.width_mlp is not None
        merge = (
            None
            if input_shape[0] == output_shape[0]
            else input_shape[0] // output_shape[0]
        )
        model = _TransformerTorchModifiable(
            ntoken=input_shape[0],
            d_model=self.emb_dim,
            nhead=self.num_heads,
            d_hid=self.width_mlp,
            nlayers=self.depth,
            dropout=self.drop_out,
            layer_norm=self.layer_norm,
            pos_encoding=self.pos_encoding,
            linear=need_linear,
            merge=merge,
        )
        self.init.initialize(model)
        return model


@dataclass
class FreezableTransformerModel(BasicTransformerModel):
    freeze_attn: bool = False
    freeze_linear: bool = False
    freeze_emb: bool = False

    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> nn.Module:
        need_linear = True if self.width_mlp else False
        model = _TransformerTorchFreezable(
            ntoken=input_shape[0],
            d_model=self.emb_dim,
            nhead=self.num_heads,
            d_hid=self.width_mlp,
            nlayers=self.depth,
            dropout=self.drop_out,
            pos_encoding=self.pos_encoding,
            layer_norm=self.layer_norm,
            linear=need_linear,
            freeze_attn=self.freeze_attn,
            freeze_linear=self.freeze_linear,
            freeze_emb=self.freeze_emb,
        )
        self.init.initialize(model)
        return model


@dataclass
class SentimentTransformerModel(TransformerModel):
    def load_model(
        self, input_shape: np.ndarray, output_shape: np.ndarray
    ) -> nn.Module:
        model = _SentimentTransformerTorch(
            ntoken=input_shape[0],
            d_model=self.emb_dim,
            nhead=self.num_heads,
            d_hid=self.width_mlp,
            nlayers=self.depth,
            dropout=self.drop_out,
            num_outputs=output_shape[0],
        )
        self.init.initialize(model)
        return model


class _TransformerTorch(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        merge: int | None = None,
    ):
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model
        self.dropout = dropout
        self.nhead = nhead
        self.d_hid = d_hid
        self.n_layers = nlayers
        self.merge = merge

        self.embedding = nn.Embedding(self.ntoken, self.d_model)
        self.pos_encoder = self.get_pos_encoding()
        self.transformer_encoder = self.get_encoder()
        self.classifier_head = nn.Linear(d_model, ntoken)

        if self.merge:
            self.pad = self.ntoken % self.merge != 0
            if self.pad:
                pad_amount = self.ntoken % self.merge
                self.padding = nn.ConstantPad1d((0, pad_amount), value=0)
            self.class_merger = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.merge,
                stride=self.merge,
                padding=0,
                bias=False,
            )
            nn.init.constant_(self.class_merger.weight, val=1.0)
            self.class_merger.requires_grad_(False)

    def forward_attn(self, src: Tensor) -> Tensor:
        """The attention part of the forward pass.

        Args:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        src_mask = generate_square_subsequent_mask(src.shape[0]).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer_encoder(src, src_mask)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        output = self.classifier_head(self.forward_attn(src))
        output = output.view(-1, self.ntoken)

        if self.merge:
            output = torch.unsqueeze(output, dim=1)
            if self.pad:
                output = self.padding(output)
            output = self.class_merger(output)
            output = torch.squeeze(output, dim=1)
        return output

    def get_pos_encoding(self):
        return _PositionalEncoding(self.d_model, dropout=self.dropout)

    def get_encoder(self):
        return TransformerEncoder(
            TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid, self.dropout),
            self.n_layers,
        )


class _SentimentTransformerTorch(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        num_outputs: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model

        self.embedding = nn.Embedding(ntoken, d_model)
        self.prediction_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_encoder = _PositionalEncoding(d_model, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, d_hid, dropout), nlayers
        )
        self.linear = nn.Linear(d_model, num_outputs)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        if src_mask is not None:
            raise NotImplementedError("Usage of src_mask is not implemented.")

        batch_pred_token = self.prediction_token.expand(src.shape[0], -1, -1)
        x = torch.zeros(1, src.shape[0], dtype=torch.bool, device=src.device)

        padding_mask = torch.cat(
            [x, (src == 1).transpose(0, 1)],
            dim=0,
        )

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = torch.cat([batch_pred_token, src], dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = output[:, 0]
        output = self.linear(output)
        return output


class _TransformerTorchModifiable(_TransformerTorch):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int | None,
        nlayers: int,
        layer_norm: bool,
        linear: bool,
        pos_encoding: bool,
        dropout: float = 0.5,
        merge: int | None = None,
    ):
        if d_hid is None:
            if linear:
                raise ValueError("d_hid must be specified if linear is True.")
            else:
                raise NotImplementedError(
                    "Implementation issue. "
                    "What happens when d_hid is None and linear is False is unclear."
                    "The value of d_hid is still passed to the __init__ function of the superclass."
                    "If this is the intended use, should we set d_hid to 0?"
                )

        self.pos_encoding = pos_encoding
        self.layer_norm = layer_norm
        self.linear = linear

        if not linear and d_hid is None:
            raise ValueError("d_hid must be None if linear is False.")

        super().__init__(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
            merge=merge,
        )

    def get_pos_encoding(self):
        if self.pos_encoding:
            return super().get_pos_encoding()
        else:
            return torch.nn.Identity()

    def get_encoder(self):
        return TransformerEncoder(
            TransformerEncoderLayerModifiable(
                self.d_model,
                self.nhead,
                self.d_hid,
                self.dropout,
                layer_norm=self.layer_norm,
                linear=self.linear,
            ),
            self.n_layers,
        )


class _TransformerTorchFreezable(_TransformerTorchModifiable):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int | None,
        nlayers: int,
        layer_norm: bool,
        linear: bool,
        pos_encoding: bool,
        freeze_linear: bool,
        freeze_attn: bool,
        freeze_emb: bool,
        dropout: float = 0.5,
    ):
        if linear and d_hid is None:
            raise ValueError("d_hid must be specified if linear is True.")

        super().__init__(
            ntoken,
            d_model,
            nhead,
            d_hid,
            nlayers,
            layer_norm=layer_norm,
            linear=linear,
            pos_encoding=pos_encoding,
            dropout=dropout,
        )
        if freeze_linear:
            self.classifier_head.requires_grad_(False)
        if freeze_emb:
            self.embedding.requires_grad_(False)
        if freeze_attn:
            self.transformer_encoder.requires_grad_(False)


class TransformerEncoderLayerModifiable(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layer_norm: bool = True,
        linear: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayerModifiable, self).__init__()
        self.layer_norm = layer_norm
        self.linear = linear
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        if linear:
            if dim_feedforward is None:
                raise ValueError("dim_feedforward should be specified if linear=True")
            self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
            self.dropout2 = nn.Dropout(dropout)

        self.norm_first = norm_first
        if layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayerModifiable, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        x = src
        if self.norm_first:
            if self.layer_norm:
                x = self.norm1(x)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.layer_norm:
                x = self.norm2(x)
            if self.linear:
                x = x + self._ff_block(x)
        else:
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.layer_norm:
                x = self.norm1(x)
            if self.linear:
                x = x + self._ff_block(x)
            if self.layer_norm:
                x = self.norm2(x)

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class _GPTModifiable(_TransformerTorch):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int | None,
        nlayers: int,
        layer_norm: bool = True,
        layer_norm_eps: float | None = 0.00001,
        linear: bool = True,
        pos_encoding: bool = True,
        attn_dropout: float = 0.1,
        embd_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        merge: int | None = None,
    ):
        if d_hid is None:
            d_hid = d_model * 4

        self.pos_encoding = pos_encoding
        self.layer_norm = layer_norm
        self.linear = linear

        self.layer_norm_eps = layer_norm_eps

        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.embd_dropout = embd_dropout

        super().__init__(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            merge=merge,
        )

    def get_pos_encoding(self):
        if self.pos_encoding:
            return _PositionalEncoding(self.d_model, dropout=self.embd_dropout)
        else:
            return torch.nn.Identity()

    def get_encoder(self):
        final_ln = nn.LayerNorm(self.d_model, eps=self.layer_norm_eps)
        return TransformerEncoder(
            GPTEncoderLayerModifiable(
                self.d_model,
                self.nhead,
                self.d_hid,
                self.attn_dropout,
                self.resid_dropout,
                layer_norm=self.layer_norm,
                linear=self.linear,
            ),
            self.n_layers,
            norm=final_ln,
        )


class GPTEncoderLayerModifiable(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int | None,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        # GPT2 norms pre-transform
        norm_first: bool = True,
        device=None,
        dtype=None,
        layer_norm: bool = True,
        linear: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(GPTEncoderLayerModifiable, self).__init__()
        self.layer_norm = layer_norm
        self.linear = linear
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            batch_first=batch_first,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        if linear:
            if dim_feedforward is None:
                raise ValueError("dim_feedforward should be specified if linear=True")
            self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
            self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        # Use post attention and post ff, attention drop out is internal to mha
        self._resid_dropout = nn.Dropout(resid_dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(GPTEncoderLayerModifiable, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        x = src
        if self.norm_first:
            if self.layer_norm:
                x = self.norm1(x)
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.layer_norm:
                x = self.norm2(x)
            if self.linear:
                x = x + self._ff_block(x)
        else:
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            if self.layer_norm:
                x = self.norm1(x)
            if self.linear:
                x = x + self._ff_block(x)
            if self.layer_norm:
                x = self.norm2(x)

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self._resid_dropout(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        if self.activation == F.gelu:
            # most gpt implementations use approximate gelu
            x = self.linear2(F.gelu(self.linear1(x), approximate="tanh"))
        else:
            x = self.linear2(self.activation(self.linear1(x)))
        return self._resid_dropout(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        assert isinstance(self.pe, Tensor)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask
