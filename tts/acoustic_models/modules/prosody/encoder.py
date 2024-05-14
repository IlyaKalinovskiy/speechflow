import torch

from tts.acoustic_models.modules.common.length_regulators import LengthRegulator
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentOutput, EncoderOutput
from tts.acoustic_models.modules.prosody.phoneme_encoder import (
    PhonemeEncoder,
    PhonemeEncoderParams,
)
from tts.acoustic_models.modules.prosody.vq_encoder import VQEncoder, VQEncoderParams

__all__ = ["ProsodyEncoder", "ProsodyEncoderParams"]


class ProsodyEncoderParams(VQEncoderParams, PhonemeEncoderParams):
    pass


class ProsodyEncoder(Component):
    params: ProsodyEncoderParams

    def __init__(self, params: ProsodyEncoderParams, input_dim: int):
        super().__init__(params, input_dim)

        self.encoder = VQEncoder(params, input_dim)
        self.adaptor_encoder = VQEncoder(params, input_dim)
        self.phoneme_encoder = PhonemeEncoder(params, params.token_emb_dim)
        self.length_regulator = LengthRegulator()

    @property
    def output_dim(self):
        return self.encoder.output_dim + self.phoneme_encoder.output_dim

    def forward_step(self, x: ComponentOutput) -> EncoderOutput:
        phon_emb = self.phoneme_encoder(x)
        durations = x.model_inputs.word_lengths
        encoder_output = self.encoder(x)
        adaptor_encoder_output = self.adaptor_encoder(x)
        content = [encoder_output.content, adaptor_encoder_output.content]
        content_lens = [
            x.model_inputs.transcription_lengths,
            x.model_inputs.transcription_lengths,
        ]

        for i, c in enumerate(content):
            c = self.length_regulator(
                c, durations, max(x.model_inputs.transcription_lengths)
            )[0]
            content[i] = torch.cat([c, phon_emb.content], dim=2)

        return EncoderOutput.copy_from(x).set_content(content, content_lens)
