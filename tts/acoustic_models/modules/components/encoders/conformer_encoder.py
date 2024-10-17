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
    n_layers: int = 4
    kernel_size: int = 31
    p_dropout: float = 0.1
    use_pe: bool = True


class ConformerEncoder(CNNEncoder):
    params: ConformerEncoderParams

    def __init__(self, params: ConformerEncoderParams, input_dim):
        super().__init__(params, input_dim)

        self.prenet_layer = ConvPrenet(
            in_channels=super().output_dim,
            out_channels=params.encoder_inner_dim,
        )

        if params.use_pe:
            self.pe = PositionalEncoding(
                params.encoder_inner_dim, max_len=params.max_input_length
            )

        self.encoder = Conformer(
            input_dim=params.encoder_inner_dim,
            num_heads=params.n_heads,
            ffn_dim=params.encoder_inner_dim,
            num_layers=params.n_layers,
            depthwise_conv_kernel_size=params.kernel_size,
            dropout=params.p_dropout,
        )

        self.proj = Regression(params.encoder_inner_dim, self.params.encoder_output_dim)

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        x = self.prenet_layer(x.transpose(1, -1)).transpose(1, -1)

        if self.params.use_pe:
            x = self.pe(x)

        z, _ = self.encoder(x, x_lens)
        y = self.proj(z)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(z)
