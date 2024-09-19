import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import (
    get_mask_from_lengths,
    get_sinusoid_encoding_table,
    run_rnn_on_padded_sequence,
)
from tts.acoustic_models.modules.ada_speech.modules import FFTBlock
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
    max_output_length: int = 0
    head: int = 2
    conv_filter_size: int = 1024
    conv_kernel_size: tp.Tuple[int, int] = (9, 1)
    dropout: float = 0.2
    cln: bool = True
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    with_mix_style: bool = False


class AdaDecoder(Component):
    """Decoder from AdaSpeech paper https://github.com/tuanh123789/AdaSpeech/tree/main."""

    params: AdaDecoderParams

    def __init__(self, params: AdaDecoderParams, input_dim):
        super().__init__(params, input_dim)

        n_position = params.max_output_length + 1
        d_word_vec = params.decoder_inner_dim
        n_layers = params.decoder_num_layers
        n_head = params.head
        d_k = d_v = params.decoder_inner_dim // params.head
        d_model = params.decoder_inner_dim
        d_inner = params.conv_filter_size
        kernel_size = params.conv_kernel_size
        dropout = params.dropout
        cln = params.cln

        if not params.condition:
            cln = False

        self.max_seq_len = params.max_output_length
        self.d_model = d_model

        self.feat_proj = nn.Linear(input_dim, params.decoder_inner_dim)

        if params.decoder_inner_dim == 256:
            self.condition_proj = nn.Linear(
                params.condition_dim, params.decoder_inner_dim
            )
        elif params.decoder_inner_dim == 512:
            self.condition_proj = nn.Linear(
                params.condition_dim, params.decoder_inner_dim // params.head
            )
        else:
            if cln:
                raise NotImplementedError

        self.output_proj = nn.Linear(params.decoder_inner_dim, params.decoder_output_dim)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.dec_layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_v,
                    d_inner,
                    kernel_size,
                    cln=cln,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.ms_layer_stack = nn.ModuleList(
            [
                MixStyle() if params.with_mix_style else nn.Identity()
                for _ in range(n_layers)
            ]
        )

        self.gate_layer = nn.Linear(
            d_model,
            1,
            bias=True,
        )

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]

        if self.params.condition:
            g = self.condition_proj(
                self.get_condition(inputs, self.params.condition)
            ).squeeze(1)
        else:
            g = None

        enc_seq = self.feat_proj(content)
        mask = get_mask_from_lengths(content_lengths)

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer, ms_layer in zip(self.dec_layer_stack, self.ms_layer_stack):
            dec_output, dec_slf_attn = dec_layer(
                dec_output, g, mask=mask, slf_attn_mask=slf_attn_mask
            )
            dec_output = ms_layer(dec_output)

        y = self.output_proj(dec_output)
        gate = self.gate_layer(dec_output)

        inputs.additional_content["fft_output"] = dec_output

        outputs = DecoderOutput.copy_from(inputs)
        outputs.set_content(y, inputs.content_lengths)
        outputs.gate = gate
        return outputs


class AdaDecoderWithRNNParams(AdaDecoderParams):
    rnn_cell: str = "lstm"
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_condition: tp.Tuple[str, ...] = ()
    rnn_condition_dim: int = 0


class AdaDecoderWithRNN(AdaDecoder):
    params: AdaDecoderWithRNNParams

    def __init__(self, params: AdaDecoderWithRNNParams, input_dim):
        super().__init__(params, input_dim)

        if params.rnn_cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=params.decoder_inner_dim + params.rnn_condition_dim,
                hidden_size=params.decoder_inner_dim,
                num_layers=params.rnn_num_layers,
                bidirectional=params.rnn_bidirectional,
            )
        else:
            raise NotImplementedError

        self.linear = nn.Sequential(
            nn.Linear(
                (1 + params.rnn_bidirectional) * params.decoder_inner_dim,
                params.decoder_inner_dim,
            ),
            nn.ReLU(),
            nn.Linear(params.decoder_inner_dim, params.decoder_output_dim),
        )

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        result: DecoderOutput = super().forward_step(inputs)
        x = result.additional_content["fft_output"]

        if self.params.rnn_condition:
            g = self.get_condition(inputs, self.params.rnn_condition)
            g = g.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, g], dim=2)

        after_rnn = run_rnn_on_padded_sequence(self.rnn, x, result.content_lengths)
        spec = self.linear(after_rnn)

        outputs = DecoderOutput.copy_from(result)
        outputs.set_content([result.content, spec], result.content_lengths)
        outputs.gate = result.gate
        return outputs
