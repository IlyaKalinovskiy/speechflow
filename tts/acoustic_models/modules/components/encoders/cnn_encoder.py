from torch import nn

from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.components.encoders.ling_condition import (
    LinguisticCondition,
    LinguisticConditionParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["CNNEncoder", "CNNEncoderParams"]


class CNNEncoderParams(LinguisticConditionParams):
    cnn_n_layers: int = 3
    cnn_n_channels: int = 256
    cnn_kernel_size: int = 3


class CNNEncoder(LinguisticCondition):
    params: CNNEncoderParams

    def __init__(self, params: CNNEncoderParams, input_dim):
        super().__init__(params, input_dim)

        self.convolutions = nn.ModuleList()
        for idx in range(params.cnn_n_layers):
            in_dim = input_dim if idx == 0 else params.cnn_n_channels
            out_dim = (
                params.cnn_n_channels if idx + 1 != params.cnn_n_layers else input_dim
            )
            conv_layer = nn.Sequential(
                Conv(
                    in_dim,
                    out_dim,
                    kernel_size=params.cnn_kernel_size,
                    stride=1,
                    padding=int((params.cnn_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
            )
            self.convolutions.append(conv_layer)

    @property
    def output_dim(self):
        return super().output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        if self.params.cnn_n_layers:
            x = x.transpose(1, -1)
            for conv in self.convolutions:
                x = conv(x)

            x = x.transpose(1, -1)

        y = super().add_ling_features(x, inputs)

        return EncoderOutput.copy_from(inputs).set_content(y)
