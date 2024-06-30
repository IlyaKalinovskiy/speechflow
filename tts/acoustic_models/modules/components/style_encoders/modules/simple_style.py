from torch import nn

from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.components.style_encoders.style_encoder import (
    StyleEncoderParams,
)

__all__ = [
    "SimpleStyle",
    "SimpleStyleParams",
]


class SimpleStyleParams(StyleEncoderParams):
    pass


class SimpleStyle(Component):
    params: SimpleStyleParams

    def __init__(self, params: SimpleStyleParams, input_dim: int):
        super().__init__(params, input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, params.vp_output_dim),
        )

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def encode(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        return self.proj(x.squeeze(-1))

    def forward_step(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        style_emb = self.encode(x, x_lengths, model_inputs, **kwargs)
        return style_emb, {}, {}
