import typing as tp

from torch import nn

from speechflow.training.utils.tensor_utils import (
    get_mask_from_lengths,
    get_sinusoid_encoding_table,
)
from tts.acoustic_models.modules.ada_speech.modules import FFTBlock
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["AdaEncoder", "AdaEncoderParams"]


class AdaEncoderParams(EncoderParams):
    layer: int = 4
    head: int = 2
    conv_filter_size: int = 1024
    conv_kernel_size: tp.Tuple[int, int] = (9, 1)
    dropout: float = 0.2
    cln: bool = True
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0


class AdaEncoder(Component):
    """Encoder from AdaSpeech paper https://github.com/tuanh123789/AdaSpeech/tree/main."""

    params: AdaEncoderParams

    def __init__(self, params: AdaEncoderParams, input_dim):
        super().__init__(params, input_dim)

        n_position = params.max_input_length + 1
        e_word_vec = params.encoder_inner_dim
        n_layers = params.layer
        n_head = params.head
        e_k = e_v = params.encoder_inner_dim // params.head
        e_model = params.encoder_inner_dim
        e_inner = params.conv_filter_size
        kernel_size = params.conv_kernel_size
        dropout = params.dropout
        cln = params.cln

        if not params.condition:
            cln = False

        self.max_seq_len = params.max_input_length
        self.e_model = e_model

        self.feat_proj = nn.Linear(input_dim, params.encoder_inner_dim)

        if params.encoder_inner_dim == 256:
            self.condition_proj = nn.Linear(
                params.condition_dim, params.encoder_inner_dim
            )
        elif params.encoder_inner_dim == 512:
            self.condition_proj = nn.Linear(
                params.condition_dim, params.encoder_inner_dim // params.head
            )
        else:
            if cln:
                raise NotImplementedError
        self.output_proj = nn.Linear(params.encoder_inner_dim, params.encoder_output_dim)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, e_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    e_model,
                    n_head,
                    e_k,
                    e_v,
                    e_inner,
                    kernel_size,
                    cln=cln,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        content = self.get_content(inputs)[0]
        content_lengths = self.get_content_lengths(inputs)[0]

        if self.params.condition:
            g = self.condition_proj(self.get_condition(inputs, self.params.condition))
        else:
            g = None

        enc_seq = self.feat_proj(content)
        mask = get_mask_from_lengths(content_lengths)

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.e_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            enc_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, g, mask=mask, slf_attn_mask=slf_attn_mask
            )

        y = self.output_proj(enc_output)
        return EncoderOutput.copy_from(inputs).set_content(y)
