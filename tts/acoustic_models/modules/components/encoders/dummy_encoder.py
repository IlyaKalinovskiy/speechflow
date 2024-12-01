from torch import nn

from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["DummyEncoder", "DummyEncoderParams"]


class DummyEncoderParams(EncoderParams):
    # projection
    use_projection: bool = False
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class DummyEncoder(Component):
    params: DummyEncoderParams

    def __init__(self, params, input_dim):
        super().__init__(params, input_dim)

        if params.use_projection and input_dim != params.encoder_output_dim:
            self.proj = Regression(
                input_dim,
                params.encoder_output_dim,
                p_dropout=params.projection_p_dropout,
                activation_fn=params.projection_activation_fn,
            )
        else:
            if input_dim != self.params.encoder_output_dim:
                raise RuntimeError(
                    "Dimension of output shape must match dimension of input shape"
                )
            self.proj = nn.Identity()

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        out = EncoderOutput.copy_from(inputs)

        content = self.get_content(inputs)

        for idx in range(len(content)):
            content[idx] = self.proj(content[idx])

        return out.set_content(content)
