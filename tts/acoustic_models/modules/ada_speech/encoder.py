import typing as tp

from torch import nn

from speechflow.utils.tensor_utils import apply_mask, get_sinusoid_encoding_table
from tts.acoustic_models.modules.ada_speech.modules import FFTBlock
from tts.acoustic_models.modules.common.blocks import ConvPrenet, Regression
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["AdaEncoder", "AdaEncoderParams"]


class AdaEncoderParams(CNNEncoderParams):
    n_heads: int = 2
    conv_filter_size: int = 1024
    conv_kernel_sizes: tp.Tuple[int, int] = (9, 1)
    p_dropout: float = 0.1

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class AdaEncoder(CNNEncoder):
    """Encoder from AdaSpeech paper https://github.com/tuanh123789/AdaSpeech/tree/main."""

    params: AdaEncoderParams

    def __init__(self, params: AdaEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim
        inner_dim = params.encoder_inner_dim

        n_position = params.max_input_length + 1
        n_layers = params.encoder_num_layers
        n_heads = params.n_heads
        e_k = e_v = inner_dim // params.n_heads
        e_inner = params.conv_filter_size
        kernel_size = params.conv_kernel_sizes

        self.prenet = ConvPrenet(in_dim, inner_dim)

        self.pe = nn.Parameter(
            get_sinusoid_encoding_table(n_position, inner_dim).unsqueeze(0),
            requires_grad=False,
        )

        self.enc_layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    inner_dim,
                    n_heads,
                    e_k,
                    e_v,
                    e_inner,
                    kernel_size,
                    c_dim=params.condition_dim,
                    dropout=params.p_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        if params.use_projection:
            self.proj = Regression(
                inner_dim,
                params.encoder_output_dim,
                p_dropout=params.projection_p_dropout,
                activation_fn=params.projection_activation_fn,
            )
        else:
            self.proj = apply_mask

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = inputs.get_content_and_mask()
        c = self.get_condition(inputs, self.params.condition)

        enc_seq = self.prenet(x.transpose(1, -1)).transpose(1, -1)

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.params.max_input_length:
            # -- Prepare masks
            slf_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.params.encoder_inner_dim
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.params.max_input_length)

            # -- Prepare masks
            slf_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_output = enc_seq[:, :max_len, :] + self.pe[:, :max_len, :].expand(
                batch_size, -1, -1
            )
            x_mask = x_mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for enc_layer in self.enc_layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, c, mask=x_mask, slf_attn_mask=slf_attn_mask
            )

        y = self.proj(enc_output, x_mask)

        return EncoderOutput.copy_from(inputs).set_content(y)
