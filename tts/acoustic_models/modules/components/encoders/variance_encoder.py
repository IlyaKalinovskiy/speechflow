import typing as tp

import torch

from torch import nn

from speechflow.training.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.layers import Conv, LearnableSwish
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["VarianceEncoder", "VarianceEncoderParams"]


class VarianceEncoderParams(EncoderParams):
    kernel_sizes: tp.Tuple[int, ...] = (3, 7, 13, 3)
    bidirectional: bool = True
    p_dropout: float = 0.1


class VarianceEncoder(Component):
    params: VarianceEncoderParams

    def __init__(self, params: VarianceEncoderParams, input_dim):
        super().__init__(params, input_dim)

        rnn_dim = params.encoder_inner_dim // (params.bidirectional + 1)
        num_rnn_layers = params.encoder_num_layers
        filter_size = params.encoder_inner_dim
        first_convs_kernel_sizes = params.kernel_sizes[:-1]
        second_convs_kernel_sizes = params.kernel_sizes[-1]
        dropout = params.p_dropout

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
                    nn.Dropout(dropout),
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
            nn.Dropout(dropout),
        )

        self.rnn = nn.LSTM(
            params.encoder_inner_dim,
            rnn_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=params.bidirectional,
        )

        self.proj = Regression(rnn_dim * 2, self.params.encoder_output_dim)

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

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

        outputs = EncoderOutput.copy_from(inputs)
        outputs = outputs.set_content(y)
        outputs.hidden_state = x
        return outputs
