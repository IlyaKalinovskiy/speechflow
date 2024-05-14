import torch.nn.functional as F

from torch import nn

from speechflow.training.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["PhonemeEncoder", "PhonemeEncoderParams"]


class PhonemeEncoderParams(EncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5


class PhonemeEncoder(Component):
    params: PhonemeEncoderParams

    def __init__(self, params: PhonemeEncoderParams, input_dim):
        super().__init__(params, input_dim)

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
            input_dim,  # type: ignore
            params.encoder_inner_dim,
            params.encoder_num_layers,
            batch_first=True,
            bidirectional=True,
        )

    @property
    def output_dim(self):
        return self.params.encoder_inner_dim * 2

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x = inputs.embeddings["phoneme_embeddings"].transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))

        output = run_rnn_on_padded_sequence(
            self.lstm, x.transpose(1, 2), inputs.model_inputs.text_lengths
        )

        return EncoderOutput(
            content=output,
            content_lengths=inputs.content_lengths,
            model_inputs=inputs.model_inputs,
            embeddings=inputs.embeddings,
            additional_content=inputs.additional_content,
            additional_losses=inputs.additional_losses,
        )
