from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import DecoderParams

__all__ = [
    "ForwardDecoder",
    "ForwardDecoderParams",
]


class ForwardDecoderParams(DecoderParams):
    pass


class ForwardDecoder(Component):
    params: ForwardDecoderParams

    def __init__(self, params: ForwardDecoderParams, input_dim):
        super().__init__(params, input_dim)

        self.rnn = nn.LSTM(
            self.input_dim,
            params.decoder_inner_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=params.decoder_num_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(2 * params.decoder_inner_dim, params.decoder_inner_dim),
            nn.ReLU(),
            nn.Linear(params.decoder_inner_dim, params.decoder_output_dim),
        )

        self.gate_layer = nn.Linear(
            2 * params.decoder_inner_dim,
            1,
            bias=True,
        )

    @property
    def output_dim(self):
        return self.params.decoder_output_dim

    def forward_step(self, inputs: VarianceAdaptorOutput) -> DecoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        after_rnn = run_rnn_on_padded_sequence(self.rnn, x, x_lens)

        spec = self.linear(after_rnn)
        gate = self.gate_layer(after_rnn)

        outputs = DecoderOutput.copy_from(inputs).set_content(spec, x_lens)
        outputs.gate = gate
        return outputs
