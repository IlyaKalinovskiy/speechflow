from torch import nn

from tts.acoustic_models.modules.common.blocks import CBHG
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import DecoderOutput, PostnetOutput
from tts.acoustic_models.modules.params import PostnetParams

__all__ = ["ForwardPostnet", "ForwardPostnetParams"]


class ForwardPostnetParams(PostnetParams):
    highways: int = 5
    n_convolutions: int = 5
    kernel_size: int = 3


class ForwardPostnet(Component):
    params: ForwardPostnetParams

    def __init__(self, params: ForwardPostnetParams, input_dim: int):
        super().__init__(params, input_dim)
        self.cbhg = CBHG(
            in_channels=self.input_dim,
            out_channels=params.postnet_inner_dim,
            conv_banks_num=params.n_convolutions,
            highways_num=params.highways,
            kernel_size=params.kernel_size,
            bidirectional_rnn=True,
            rnn_channels=self.input_dim,
        )
        self.linear = nn.Linear(params.postnet_inner_dim, params.postnet_output_dim)

    @property
    def output_dim(self):
        return self.params.postnet_output_dim

    def forward_step(self, inputs: DecoderOutput) -> PostnetOutput:  # type: ignore
        content = self.get_content(inputs)
        x = content[-1]

        x = self.cbhg(x)

        x_out = self.linear(x)

        return PostnetOutput.copy_from(inputs).set_content(x_out)
