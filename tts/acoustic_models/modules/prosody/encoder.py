import torch

from speechflow.utils.tensor_utils import get_lengths_from_durations
from tts.acoustic_models.modules.common.length_regulators import (
    LengthRegulator,
    SoftLengthRegulator,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.encoders import RNNEncoder, RNNEncoderParams
from tts.acoustic_models.modules.components.encoders.vq_encoder import (
    VQEncoder,
    VQEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentOutput, EncoderOutput
from tts.acoustic_models.modules.prosody.transformer import MultimodalTransformerEncoder

__all__ = ["ProsodyEncoder", "ProsodyEncoderParams"]


class ProsodyEncoderParams(VQEncoderParams, RNNEncoderParams):
    mt_embed_dim: int = 1024
    mt_num_heads: int = 1
    mt_layers: int = 1


class ProsodyEncoder(Component):
    params: ProsodyEncoderParams

    def __init__(self, params: ProsodyEncoderParams, input_dim: int):
        super().__init__(params, input_dim)

        self.multimodal_transformer = MultimodalTransformerEncoder(
            embed_dim=params.mt_embed_dim,
            num_heads=params.mt_num_heads,
            layers=params.mt_layers,
        )
        self.text_encoder = RNNEncoder(params, params.token_emb_dim)
        self.vq_encoder = VQEncoder(params, input_dim)
        self.hard_lr = LengthRegulator()
        self.soft_lr = SoftLengthRegulator()

    @property
    def output_dim(self):
        return self.text_encoder.output_dim + self.vq_encoder.output_dim

    def forward_step(self, x: ComponentOutput) -> EncoderOutput:
        audio_emb = x.model_inputs.ssl_feat
        text_emb = x.model_inputs.lm_feat.transpose(0, 1)

        invert_durations = x.model_inputs.additional_inputs["word_invert_durations"]
        durations = x.model_inputs.additional_inputs["word_durations"]
        audio_emb, _ = self.soft_lr(audio_emb, invert_durations, durations.shape[1])

        bimodal_embs = self.multimodal_transformer(
            audio_emb.transpose(0, 1), text_emb, text_emb
        ).transpose(0, 1)
        x.set_content(bimodal_embs, get_lengths_from_durations(invert_durations))

        text = x.embeddings["transcription"]
        text_lens = x.model_inputs.transcription_lengths
        text_embs, _ = self.text_encoder.process_content(text, text_lens, x.model_inputs)

        encoder_output = self.vq_encoder(x)

        quant = encoder_output.content
        dura = x.model_inputs.word_lengths
        quant, _ = self.hard_lr(quant, dura, text_embs.shape[1])
        context = torch.cat([quant, text_embs], dim=-1)

        return EncoderOutput.copy_from(x).set_content(context, text_lens)
