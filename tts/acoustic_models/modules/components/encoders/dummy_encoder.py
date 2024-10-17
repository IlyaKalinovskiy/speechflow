from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = ["DummyEncoder", "DummyEncoderParams"]


class DummyEncoderParams(EncoderParams):
    pass


class DummyEncoder(Component):
    params: DummyEncoderParams

    def __init__(self, params, input_dim):
        super().__init__(params, input_dim)

    @property
    def output_dim(self):
        return self.input_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        return inputs  # type: ignore
