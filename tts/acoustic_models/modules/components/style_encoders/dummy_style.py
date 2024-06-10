from tts.acoustic_models.modules.component import Component
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

    def encode(self, x, x_mask, **kwargs):
        return x

    def forward_step(self, x, x_mask, **kwargs):
        return self.encode(x, x_mask, **kwargs), {}, {}
