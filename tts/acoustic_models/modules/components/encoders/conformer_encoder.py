from torch import nn
from torchaudio.models import Conformer

from tts.acoustic_models.modules.common.blocks import ConvPrenet, Regression
from tts.acoustic_models.modules.common.pos_encoder import PositionalEncoding
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["ConformerEncoder", "ConformerEncoderParams"]


class ConformerEncoderParams(CNNEncoderParams):
    n_heads: int = 4
    kernel_size: int = 31
    p_dropout: float = 0.1
    use_pe: bool = True

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class ConformerEncoder(CNNEncoder):
    params: ConformerEncoderParams

    def __init__(self, params: ConformerEncoderParams, input_dim):
        super().__init__(params, input_dim)

        inner_dim = params.encoder_inner_dim

        self.prenet_layer = ConvPrenet(
            in_channels=super().output_dim,
            out_channels=inner_dim,
        )

        if params.use_pe:
            self.pe = PositionalEncoding(inner_dim, max_len=params.max_input_length)

        self.encoder = Conformer(
            input_dim=inner_dim,
            num_heads=params.n_heads,
            ffn_dim=inner_dim,
            num_layers=params.encoder_num_layers,
            depthwise_conv_kernel_size=params.kernel_size,
            dropout=params.p_dropout,
        )

        if params.use_projection and inner_dim != params.encoder_output_dim:
            self.proj = Regression(
                inner_dim,
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
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        x = self.prenet_layer(x.transpose(1, -1)).transpose(1, -1)

        if self.params.use_pe:
            x = self.pe(x)

        z, _ = self.encoder(x, x_lens)
        y = self.proj(z)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(z)
