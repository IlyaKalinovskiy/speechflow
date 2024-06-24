from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.components.style_encoders.style_encoder import (
    StyleEncoderParams,
)

__all__ = ["DummyStyle", "DummyStyleParams"]


class DummyStyleParams(StyleEncoderParams):
    pass


class DummyStyle(Component):
    params: DummyStyleParams

    def __init__(self, params: DummyStyleParams, input_dim: int):
        super().__init__(params, input_dim)

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def encode(self, x, x_lengths, model_inputs, **kwargs):
        return x

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        return self.encode(x, x_lengths, model_inputs, **kwargs), {}, {}
