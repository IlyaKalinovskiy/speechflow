import torch
import torch.nn.functional as F

from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.common.vector_quantizer import (
    VectorQuantizer,
    VectorQuantizerOutput,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["VQEncoder", "VQEncoderParams"]


class VQEncoderParams(EncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5
    vq_embedding_dim: int = 128
    vq_codebook_size: int = 256


class VQEncoder(Component):
    params: VQEncoderParams

    def __init__(self, params: VQEncoderParams, input_dim, level="token"):
        super().__init__(params, input_dim)
        self._level = level

        convolutions = []
        for _ in range(params.n_convolutions):
            conv_layer = nn.Sequential(
                Conv(
                    input_dim,
                    input_dim,
                    kernel_size=params.kernel_size,
                    stride=1,
                    padding=int((params.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(input_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            input_dim,
            params.encoder_inner_dim,
            params.encoder_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.pre_vq_conv = nn.Conv1d(
            in_channels=params.encoder_inner_dim * 2,
            out_channels=params.vq_embedding_dim,
            kernel_size=3,
            padding=1,
        )

        self.vq = VectorQuantizer(
            embedding_dim=params.vq_embedding_dim,
            codebook_size=params.vq_codebook_size,
        )

    @property
    def output_dim(self):
        return self.params.vq_embedding_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        content = (
            [inputs.content] if not isinstance(inputs.content, list) else inputs.content
        )
        x = content[0]

        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))

        if self._level == "token":
            lengths = torch.count_nonzero(inputs.model_inputs.word_lengths, dim=1)
        elif self._level == "phoneme":
            lengths = inputs.model_inputs.text_lengths
        else:
            lengths = inputs.model_inputs.input_lengths
        y = run_rnn_on_padded_sequence(self.lstm, x.transpose(1, 2), lengths)

        z = self.pre_vq_conv(y.transpose(1, 2))

        vq_output: VectorQuantizerOutput = self.vq(z)  # type: ignore
        assert not isinstance(vq_output.content, list)

        for k, v in vq_output.additional_content.items():
            inputs.additional_content[f"{k}_encoder_{self.id}"] = v

        for k, v in vq_output.additional_losses.items():
            inputs.additional_losses[f"{k}_encoder_{self.id}"] = v

        return EncoderOutput.copy_from(inputs).set_content(
            vq_output.content.transpose(1, 2)
        )
