import math
import typing as tp

import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils import weight_norm

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["AdainEncoder", "AdainEncoderParams"]


class AdainEncoderParams(CNNEncoderParams):
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Literal["cat", "adanorm"] = "cat"
    bidirectional: bool = True
    p_dropout: float = 0.1
    upsample_x2: bool = False


class AdainEncoder(CNNEncoder):
    params: AdainEncoderParams

    def __init__(self, params: AdainEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim
        inner_dim = params.encoder_inner_dim

        self.rnn = nn.LSTM(
            in_dim,
            inner_dim // (params.bidirectional + 1),
            num_layers=params.encoder_num_layers,
            batch_first=True,
            bidirectional=params.bidirectional,
            dropout=params.p_dropout,
        )

        if params.condition_dim > 0:
            self.adain = nn.ModuleList()
            self.adain.append(
                AdainResBlk1d(
                    inner_dim, inner_dim, params.condition_dim, dropout_p=params.p_dropout
                )
            )
            self.adain.append(
                AdainResBlk1d(
                    inner_dim,
                    inner_dim,
                    params.condition_dim,
                    upsample=params.upsample_x2,
                    dropout_p=params.p_dropout,
                )
            )
            self.adain.append(
                AdainResBlk1d(
                    inner_dim, inner_dim, params.condition_dim, dropout_p=params.p_dropout
                )
            )
        else:
            self.adain = None

        self.proj = Regression(inner_dim, self.params.encoder_output_dim)

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        inputs = super().forward_step(inputs)

        cond = self.get_condition(inputs, self.params.condition)
        if cond is not None:
            cond = cond.squeeze(1)

        x, x_lens, x_mask = self.get_content_and_mask(inputs)
        x = run_rnn_on_padded_sequence(self.rnn, x, x_lens)

        z = x.transpose(1, -1)

        if self.adain is not None:
            for block in self.adain:
                z = block(z, cond)

        y = self.proj(z.transpose(1, -1))

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)


class AdaIN1d(nn.Module):
    def __init__(self, condition_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(condition_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        condition_dim=64,
        actv=nn.LeakyReLU(0.2),
        upsample=False,
        dropout_p=0.0,
    ):
        super().__init__()
        self.actv = actv
        self.upsample = UpSample1d() if upsample else nn.Identity()
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, condition_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )
        else:
            self.pool = nn.Identity()

    def _build_weights(self, dim_in, dim_out, condition_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(condition_dim, dim_in)
        self.norm2 = AdaIN1d(condition_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
