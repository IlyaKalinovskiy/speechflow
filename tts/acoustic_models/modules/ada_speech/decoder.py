import typing as tp

from torch import nn

from speechflow.utils.tensor_utils import (
    apply_mask,
    get_sinusoid_encoding_table,
    run_rnn_on_padded_sequence,
)
from tts.acoustic_models.modules.ada_speech.modules import FFTBlock
from tts.acoustic_models.modules.common.blocks import ConvPrenet
from tts.acoustic_models.modules.common.mixstyle import MixStyle
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "AdaDecoder",
    "AdaDecoderParams",
    "AdaDecoderWithRNN",
    "AdaDecoderWithRNNParams",
]


class AdaDecoderParams(DecoderParams):
    n_heads: int = 2
    conv_filter_size: int = 1024
    conv_kernel_sizes: tp.Tuple[int, int] = (9, 1)
    p_dropout: float = 0.2
    with_mix_style: bool = True

    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0


class AdaDecoder(Component):
    """Decoder from AdaSpeech paper https://github.com/tuanh123789/AdaSpeech/tree/main."""

    params: AdaDecoderParams

    def __init__(self, params: AdaDecoderParams, input_dim):
        super().__init__(params, input_dim)

        inner_dim = params.decoder_inner_dim

        n_position = params.max_output_length + 1
        n_layers = params.decoder_num_layers
        n_heads = params.n_heads
        d_k = d_v = inner_dim // params.n_heads
        d_model = inner_dim
        d_inner = params.conv_filter_size
        kernel_size = params.conv_kernel_sizes

        self.prenet = ConvPrenet(input_dim, inner_dim)

        self.pe = nn.Parameter(
            get_sinusoid_encoding_table(n_position, inner_dim).unsqueeze(0),
            requires_grad=False,
        )

        self.dec_layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    inner_dim,
                    n_heads,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size,
                    c_dim=params.condition_dim,
                    dropout=params.p_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.proj = nn.Linear(inner_dim, params.decoder_output_dim)

        self.ms_layer_stack = nn.ModuleList(
            [
                MixStyle() if params.with_mix_style else nn.Identity()
                for _ in range(n_layers)
            ]
        )

        self.gate_layer = nn.Linear(d_model, 1)

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = inputs.get_content_and_mask()
        c = self.get_condition(inputs, self.params.condition)

        enc_seq = self.prenet(x.transpose(1, -1)).transpose(1, -1)

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.params.max_output_length:
            # -- Prepare masks
            slf_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.params.decoder_inner_dim
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.params.max_output_length)

            # -- Prepare masks
            slf_attn_mask = x_mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.pe[:, :max_len, :].expand(
                batch_size, -1, -1
            )
            x_mask = x_mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer, ms_layer in zip(self.dec_layer_stack, self.ms_layer_stack):
            dec_output, dec_slf_attn = dec_layer(
                dec_output, c, mask=x_mask, slf_attn_mask=slf_attn_mask
            )
            dec_output = ms_layer(dec_output)

        y = apply_mask(self.proj(dec_output), x_mask)
        gate = self.gate_layer(dec_output)

        inputs.additional_content["fft_output"] = dec_output

        outputs = DecoderOutput.copy_from(inputs)
        outputs.set_content(y, inputs.content_lengths)
        outputs.gate = gate
        return outputs


class AdaDecoderWithRNNParams(AdaDecoderParams):
    rnn_type: tp.Literal["GRU", "LSTM"] = "LSTM"
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1


class AdaDecoderWithRNN(AdaDecoder):
    params: AdaDecoderWithRNNParams

    def __init__(self, params: AdaDecoderWithRNNParams, input_dim):
        super().__init__(params, input_dim)

        rnn_cls = getattr(nn, params.rnn_type)
        self.rnn = rnn_cls(
            params.decoder_inner_dim,
            params.decoder_inner_dim // (params.rnn_bidirectional + 1),
            num_layers=params.rnn_num_layers,
            bidirectional=params.rnn_bidirectional,
            dropout=params.rnn_p_dropout,
            batch_first=True,
        )
        self.rnn_proj = nn.Linear(params.decoder_inner_dim, params.decoder_output_dim)

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        result: DecoderOutput = super().forward_step(inputs)
        x = result.additional_content["fft_output"]
        _, x_lens, x_mask = inputs.get_content_and_mask()

        after_rnn = run_rnn_on_padded_sequence(self.rnn, x, x_lens)
        y = apply_mask(self.rnn_proj(after_rnn), x_mask)

        outputs = DecoderOutput.copy_from(result)
        outputs.set_content([result.content, y], result.content_lengths)
        outputs.gate = result.gate
        return outputs
