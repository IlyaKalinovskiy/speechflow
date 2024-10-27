import typing as tp

import torch

from tts.acoustic_models.data_types import TTSForwardInput
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentOutput
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator
from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["EmbeddingComponent"]


class EmbeddingComponent(Component):
    params: EmbeddingParams

    def __init__(
        self,
        params: EmbeddingParams,
        input_dim: tp.Optional[int] = None,
    ):
        super().__init__(params, input_dim)

        self.emb_calculator = EmbeddingCalculator(params)  # type: ignore

    @property
    def output_dim(self):
        if self.params.input == "transcription":
            return self.params.token_emb_dim
        elif self.params.input == "spectrogram":
            return self.params.spectrogram_proj_dim
        elif self.params.input == "ssl":
            return self.params.ssl_proj_dim
        else:
            raise NotImplementedError(
                f"input_embedding '{self.params.input}' is not support."
            )

    def forward_step(self, inputs: TTSForwardInput) -> ComponentOutput:  # type: ignore
        transcription = self.emb_calculator.get_transcription_embeddings(inputs)
        lm_feat = self.emb_calculator.get_lm_feat(inputs)
        plbert_feat = self.emb_calculator.get_plbert_feat(inputs)
        linear_spectrogram = self.emb_calculator.get_linear_spectrogram(inputs)
        mel_spectrogram = self.emb_calculator.get_mel_spectrogram(inputs)
        ssl_feat = self.emb_calculator.get_ssl_feat(inputs)
        ac_feat = self.emb_calculator.get_ac_feat(inputs)

        if self.params.input == "transcription":
            x = transcription
            x_lengths = inputs.transcription_lengths

        elif self.params.input == "lm_feat":
            x = lm_feat
            x_lengths = inputs.transcription_lengths

        elif "spectrogram" in self.params.input:
            x = mel_spectrogram if "mel" in self.params.input else linear_spectrogram
            x_lengths = inputs.spectrogram_lengths

        elif self.params.input == "ssl_feat":
            x = ssl_feat
            x_lengths = inputs.ssl_feat_lengths

        elif self.params.input == "ac_feat":
            x = ac_feat
            x_lengths = inputs.ac_feat_lengths

        else:
            x = None
            x_lengths = None

        ling_feat = self.emb_calculator.get_ling_feat(inputs)
        lang_emb = self.emb_calculator.get_lang_embedding(inputs)
        speaker_emb = self.emb_calculator.get_speaker_embedding(inputs)
        biometric_emb = self.emb_calculator.get_speaker_biometric_embedding(inputs)
        averages = self.emb_calculator.get_averages(inputs)
        sq_emb = self.emb_calculator.get_speech_quality_embedding(inputs)

        if lang_emb is not None and x is not None:
            x = x + lang_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        embeddings = dict(
            transcription=transcription,
            ling_feat=ling_feat,
            lm_feat=lm_feat,
            plbert_feat=plbert_feat,
            lang_emb=lang_emb,
            speaker_emb=speaker_emb,
            biometric_emb=biometric_emb,
            speech_quality_emb=sq_emb,
            ssl_feat=ssl_feat,
            ac_feat=ac_feat,
        )

        if averages is not None:
            embeddings["average_emb"] = torch.cat(list(averages.values()), dim=1)
            embeddings.update({f"average_{k}": v for k, v in averages.items()})

        embeddings = {k: v for k, v in embeddings.items() if v is not None}
        inputs.additional_inputs.update(embeddings)

        return ComponentOutput(
            content=[x],
            content_lengths=[x_lengths],
            embeddings=embeddings,
            model_inputs=inputs,
        )
