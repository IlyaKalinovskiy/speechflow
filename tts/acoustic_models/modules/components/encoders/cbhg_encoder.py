import math

import torch

from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.common.blocks import ConvPrenet, Regression
from tts.acoustic_models.modules.common.layers import Conv, HighwayNetwork
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["CBHGEncoder", "CBHGEncoderParams"]


class CBHGEncoderParams(EncoderParams):
    conv_banks_num: int = 5
    kernel_size: int = 3
    highways_num: int = 1


class CBHGEncoder(Component):
    params: CBHGEncoderParams

    def __init__(self, params: CBHGEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_channels = input_dim
        inner_channels = params.encoder_inner_dim
        out_channels = params.encoder_output_dim

        self.prenet = ConvPrenet(in_channels, inner_channels)

        self.bank_kernels = [
            params.kernel_size * (i + 1) for i in range(params.conv_banks_num)
        ]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            padding = math.ceil((k - 1) / 2)
            conv = Conv(
                inner_channels,
                inner_channels,
                kernel_size=k,
                padding=padding,
                bias=False,
                batch_norm=True,
                use_activation=True,
            )
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

        self.conv_project1 = Conv(
            params.conv_banks_num * inner_channels,
            inner_channels,
            kernel_size=params.kernel_size,
            padding=math.ceil((params.kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )
        self.conv_project2 = Conv(
            inner_channels,
            inner_channels,
            kernel_size=params.kernel_size,
            padding=math.ceil((params.kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )

        self.highways = nn.ModuleList()
        for i in range(params.highways_num):
            hn = HighwayNetwork(inner_channels)
            self.highways.append(hn)

        self.proj = Regression(inner_channels, out_channels)

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        x = self.prenet(x.transpose(2, 1))

        # Save these for later
        residual = x
        conv_bank = []

        # Convolution Bank
        for idx, conv in enumerate(self.conv1d_bank):
            c = conv(x)
            if idx % 2 == 0:
                c = F.pad(c, [0, 1])
            conv_bank.append(c)

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)

        y = self.proj(x)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
