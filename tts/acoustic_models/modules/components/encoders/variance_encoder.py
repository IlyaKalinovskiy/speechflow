import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.layers import Conv, LearnableSwish
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["VarianceEncoder", "VarianceEncoderParams"]


class VarianceEncoderParams(EncoderParams):
    # convolution block
    conv_kernel_sizes: tp.Tuple[int, ...] = (3, 7, 13, 3)
    conv_p_dropout: float = 0.1

    # rnn
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class VarianceEncoder(Component):
    params: VarianceEncoderParams

    def __init__(self, params: VarianceEncoderParams, input_dim):
        super().__init__(params, input_dim)

        filter_size = params.encoder_inner_dim
        first_convs_kernel_sizes = params.conv_kernel_sizes[:-1]
        second_convs_kernel_sizes = params.conv_kernel_sizes[-1]

        self.first_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        input_dim,
                        filter_size,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                        w_init_gain=None,
                        swap_channel_dim=True,
                    ),
                    LearnableSwish(),
                    nn.LayerNorm(filter_size),
                    nn.Dropout(params.conv_p_dropout),
                )
                for k in first_convs_kernel_sizes
            ]
        )

        self.second_conv = nn.Sequential(
            Conv(
                filter_size * len(first_convs_kernel_sizes),
                filter_size,
                kernel_size=second_convs_kernel_sizes,
                padding=(second_convs_kernel_sizes - 1) // 2,
                w_init_gain=None,
                swap_channel_dim=True,
            ),
            LearnableSwish(),
            nn.LayerNorm(filter_size),
            nn.Dropout(params.conv_p_dropout),
        )

        self.rnn = nn.LSTM(
            params.encoder_inner_dim,
            params.encoder_inner_dim // (params.rnn_bidirectional + 1),
            num_layers=params.encoder_num_layers,
            bidirectional=params.rnn_bidirectional,
            dropout=params.rnn_p_dropout,
            batch_first=True,
        )

        if (
            params.use_projection
            and params.encoder_inner_dim != params.encoder_output_dim
        ):
            self.proj = Regression(
                params.encoder_inner_dim,
                params.encoder_output_dim,
                p_dropout=params.projection_p_dropout,
                activation_fn=params.projection_activation_fn,
            )
        else:
            self.proj = nn.Identity()

    @property
    def output_dim(self):
        if self.params.use_projection:
            return self.params.encoder_output_dim
        else:
            return self.params.encoder_inner_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        after_first_conv = []
        for conv_layer in self.first_convs:
            after_first_conv.append(conv_layer(x))

        concatenated = torch.cat(after_first_conv, dim=2)
        after_second_conv = self.second_conv(concatenated)

        for conv_1 in after_first_conv:
            after_second_conv += conv_1

        x = run_rnn_on_padded_sequence(self.rnn, after_second_conv, x_lens)

        y = self.proj(x)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
