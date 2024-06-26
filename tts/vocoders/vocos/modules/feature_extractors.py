import typing as tp

import torch
import torchaudio

from torch import nn
from torch.nn import functional as F

from speechflow.data_pipeline.datasample_processors.text_processors import TextProcessor
from speechflow.training.losses.vae_loss import VAELoss
from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.batch_processor import TTSBatchProcessor
from tts.acoustic_models.modules import TTS_ENCODERS
from tts.acoustic_models.modules.additional_modules import (
    AdditionalModules,
    AdditionalModulesParams,
)
from tts.acoustic_models.modules.common.blocks import Regression, VarianceEmbedding
from tts.acoustic_models.modules.components.encoders import (
    SFEncoder,
    SFEncoderParams,
    VQEncoderWithTokenContext,
    VQEncoderWithTokenContextParams,
)
from tts.acoustic_models.modules.components.style_encoders import (
    StyleEncoder,
    StyleEncoderParams,
)
from tts.acoustic_models.modules.components.variance_predictors import (
    FrameLevelPredictorWithDiscriminator,
    FrameLevelPredictorWithDiscriminatorParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.utils.tensor_utils import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, inputs: VocoderForwardInput, **kwargs) -> torch.Tensor:
        """Extract features from the given audio.

        Args:
            inputs (VocoderForwardInput): Input audio features.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.

        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self, sample_rate=24000, n_fft=1024, hop_length=320, n_mels=80, padding="center"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(
                inputs.waveform, (pad // 2, pad // 2), mode="reflect"
            )
        else:
            audio = inputs.waveform

        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features, {}


class AudioFeatures(FeatureExtractor):
    def __init__(
        self,
        input_feat_type: tp.Literal["linear_spec", "mel_spec", "ssl_feat"] = "mel_spec",
        output_dim: int = 256,
        n_langs: int = 1,
        n_speakers: int = 2,
        speaker_emb_dim: int = 256,
        linear_spectrogram_dim: int = 513,
        mel_spectrogram_dim: int = 80,
        ssl_feat_dim: int = 1024,
        style_emb_dim: int = 192,
        condition_emb_dim: int = 32,
        encoder_type: str = "RNNEncoder",
        encoder_num_blocks: int = 1,
        encoder_num_layers: int = 1,
        encoder_inner_dim: int = 512,
        condition_type: tp.Literal["cat", "adanorm"] = "cat",
        style_encoder_type: tp.Literal[
            "SimpleStyle", "StyleSpeech", "StyleTTS2"
        ] = "StyleSpeech",
        style_feat_type: tp.Literal[
            "linear_spec", "mel_spec", "ssl_feat", "speaker_emb", "style_emb"
        ] = "mel_spec",
        style_use_gmvae: bool = False,
        style_use_fsq: bool = False,
        style_gmvae_n_components: int = 16,
        vq_type: tp.Literal["vq", "rvq", "rfsq", "rlfq"] = "rlfq",
        vq_emb_dim: int = 256,
        vq_codebook_size: int = 1024,
        vq_num_quantizers: int = 1,
        energy_interval: tp.Tuple[float, float] = (0, 150),
        pitch_interval: tp.Tuple[float, float] = (0, 850),
        average_energy_interval: tp.Tuple[float, float] = (0, 150),
        average_pitch_interval: tp.Tuple[float, float] = (0, 850),
        use_lang_emb: bool = False,
        use_speaker_emb: bool = False,
        use_speech_quality_emb: bool = False,
        use_style: bool = False,
        use_energy: bool = False,
        use_pitch: bool = False,
        use_ssl_adjustment: bool = False,
        use_vq: bool = False,
        use_averages: bool = False,
        use_range: bool = False,
        use_inverse_grad: bool = False,
        use_auxiliary_loss: bool = False,
    ):
        super().__init__()

        def _get_feat_dim(feat_name: str) -> int:
            if feat_name == "linear_spec":
                return linear_spectrogram_dim
            elif feat_name == "mel_spec":
                return mel_spectrogram_dim
            elif feat_name == "ssl_feat":
                return ssl_feat_dim
            elif style_feat_type == "speaker_emb":
                return speaker_emb_dim
            elif style_feat_type == "style_emb":
                return style_emb_dim
            else:
                raise NotImplementedError(f"feat_name '{feat_name}' is not supported")

        self.tts_bp = TTSBatchProcessor()

        self.input_feat_type = input_feat_type
        self.style_feat_type = style_feat_type

        condition = []
        condition_dim = 0

        if use_lang_emb:
            self.lang_embs = nn.Embedding(n_langs, condition_emb_dim)
            condition.append("lang_emb<no_detach>")
            condition_dim += condition_emb_dim
        else:
            self.lang_embs = None

        if use_speaker_emb:
            self.speaker_proj = nn.Linear(speaker_emb_dim, condition_emb_dim)
            condition.append("speaker_emb<no_detach>")
            condition_dim += condition_emb_dim
        else:
            self.speaker_proj = None

        if use_speech_quality_emb:
            condition.append("speech_quality_emb")
            condition_dim += 4

        if use_averages and use_energy:
            self.avr_energy_emb = VarianceEmbedding(
                interval=average_energy_interval, emb_dim=condition_emb_dim
            )
            condition.append("avr_energy_emb<no_detach>")
            condition_dim += condition_emb_dim
        else:
            self.avr_energy_emb = None

        if use_averages and use_pitch:
            self.avr_pitch_emb = VarianceEmbedding(
                interval=average_pitch_interval, emb_dim=condition_emb_dim, log_scale=True
            )
            condition.append("avr_pitch_emb<no_detach>")
            condition_dim += condition_emb_dim
        else:
            self.avr_pitch_emb = None

        if use_style:
            style_params = StyleEncoderParams(
                base_encoder_type=style_encoder_type,
                source=style_feat_type,
                source_dim=_get_feat_dim(style_feat_type),
                vp_output_dim=condition_emb_dim,
                min_spec_len=128,
                max_spec_len=512,
                use_gmvae=style_use_gmvae,
                use_fsq=style_use_fsq,
                gmvae_n_components=style_gmvae_n_components,
            )
            self.style_enc = StyleEncoder(style_params, 0)

            if style_use_gmvae:
                self.vae_scheduler = VAELoss(
                    scale=0.00002,
                    every_iter=1,
                    begin_iter=1000,
                    end_anneal_iter=10000,
                )
            else:
                self.vae_scheduler = None

            condition.append("style_emb<no_detach>")
            condition_dim += condition_emb_dim
        else:
            self.style_enc = None

        if (use_energy or use_pitch) and use_range:
            self.range_predictor = Regression(condition_emb_dim * 2, 3 * 2)
        else:
            self.range_predictor = None

        in_dim = _get_feat_dim(input_feat_type)

        # ----- init VQ encoder -----

        if use_vq:
            vq_enc_params = VQEncoderWithTokenContextParams(
                vq_type=vq_type,
                vq_codebook_size=vq_codebook_size,
                vq_num_quantizers=vq_num_quantizers,
                vq_encoder_type=encoder_type,
                vq_encoder_params={
                    "encoder_inner_dim": encoder_inner_dim,
                    "encoder_num_blocks": encoder_num_blocks,
                    "encoder_num_layers": encoder_num_layers,
                    "cnn_n_layers": 3,
                    "condition": condition,
                    "condition_dim": condition_dim,
                    "condition_type": condition_type,
                    "max_input_length": 2048 * 2,
                    "max_output_length": 2048 * 2,
                },
                encoder_output_dim=vq_emb_dim,
                tag="vq_encoder",
            )
            self.vq_enc = VQEncoderWithTokenContext(vq_enc_params, in_dim)
            in_dim = self.vq_enc.output_dim
        else:
            self.vq_enc = None

        # ----- init 1d predictors -----

        var_encoder_params = {
            "condition": condition,
            "condition_dim": condition_dim,
            "condition_type": condition_type,
        }

        if use_energy:
            energy_predictor_params = FrameLevelPredictorWithDiscriminatorParams(
                frame_encoder_type=encoder_type,
                frame_encoder_params=var_encoder_params,
                vp_hidden_channels=encoder_inner_dim,
                vp_num_layers=encoder_num_layers,
                vp_output_dim=1,
            )
            self.energy_predictor = FrameLevelPredictorWithDiscriminator(
                energy_predictor_params, in_dim
            )
        else:
            self.energy_predictor = None

        if use_pitch:
            pitch_predictor_params = FrameLevelPredictorWithDiscriminatorParams(
                frame_encoder_type=encoder_type,
                frame_encoder_params=var_encoder_params,
                vp_inner_channels=encoder_inner_dim,
                vp_num_layers=encoder_num_layers,
                vp_output_dim=1,
                use_ssl_adjustment=use_ssl_adjustment,
                ssl_feat_dim=ssl_feat_dim,
            )
            self.pitch_predictor = FrameLevelPredictorWithDiscriminator(
                pitch_predictor_params, in_dim
            )
        else:
            self.pitch_predictor = None

        # ----- init 1d source-filter encoder -----

        if use_energy or use_pitch:
            enc_params = SFEncoderParams(
                base_encoder_type=encoder_type,
                encoder_inner_dim=encoder_inner_dim,
                encoder_num_layers=encoder_num_layers,
                encoder_output_dim=output_dim,
                condition=tuple(condition),
                condition_dim=condition_dim,
                condition_type=condition_type,
                var_as_embedding=(True, True),
                var_intervals=(energy_interval, pitch_interval),
                var_log_scale=(False, True),
                max_input_length=2048 * 2,
                max_output_length=2048 * 2,
            )
            self.encoder = SFEncoder(enc_params, in_dim)
        else:
            enc_cls, enc_params_cls = TTS_ENCODERS[encoder_type]
            enc_params = enc_params_cls(
                encoder_num_blocks=encoder_num_blocks,
                encoder_num_layers=encoder_num_layers,
                encoder_inner_dim=encoder_inner_dim,
                encoder_output_dim=output_dim,
                condition=tuple(condition),
                condition_dim=condition_dim,
                condition_type=condition_type,
            )
            self.encoder = enc_cls(enc_params, _get_feat_dim(input_feat_type))

        # ----- init additional modules -----

        if use_inverse_grad:
            text_proc = TextProcessor(lang="MULTILANG")
            addm_params = AdditionalModulesParams()
            addm_params.n_symbols = text_proc.alphabet_size
            addm_params.n_symbols_per_token = text_proc.num_symbols_per_phoneme_token
            addm_params.n_speakers = n_speakers
            addm_params.speaker_emb_dim = speaker_emb_dim

            if use_vq:
                addm_params.addm_apply_token_classifier = {
                    "token_context_0": self.vq_enc.output_dim
                }
                addm_params.addm_apply_inverse_speaker_emb = {
                    "vq_encoder": self.vq_enc.output_dim
                }

            if use_style:
                addm_params.addm_apply_inverse_speaker_classifier[
                    "style_emb"
                ] = self.style_enc.output_dim

            self.addm = AdditionalModules(addm_params)
        else:
            self.addm = None

        if use_auxiliary_loss:
            enc_cls, enc_params_cls = TTS_ENCODERS[encoder_type]
            enc_params = enc_params_cls(
                encoder_num_blocks=encoder_num_blocks,
                encoder_num_layers=encoder_num_layers,
                encoder_inner_dim=encoder_inner_dim,
                encoder_output_dim=mel_spectrogram_dim,
            )
            self.mel_predictor = enc_cls(enc_params, self.encoder.output_dim)
        else:
            self.mel_predictor = None

    def _get_input_feat(
        self, inputs: VocoderForwardInput
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.input_feat_type == "linear_spec":
            return inputs.linear_spectrogram, inputs.spectrogram_lengths
        elif self.input_feat_type == "mel_spec":
            return inputs.mel_spectrogram, inputs.spectrogram_lengths
        elif self.input_feat_type == "ssl_feat":
            return inputs.ssl_feat, inputs.ssl_feat_lengths

    def _get_conditions(self, inputs: VocoderForwardInput) -> tp.Dict[str, tp.Any]:
        conditions = {}

        if self.lang_embs is not None:
            lang_id = inputs.lang_id
            conditions["lang_emb"] = self.lang_embs(lang_id)

        if self.speaker_proj is not None:
            conditions["speaker_emb"] = self.speaker_proj(inputs.speaker_emb)

        conditions["speech_quality_emb"] = inputs.speech_quality_emb

        if self.avr_energy_emb is not None:
            conditions["avr_energy_emb"] = self.avr_energy_emb(inputs.averages["energy"])

        if self.avr_pitch_emb is not None:
            conditions["avr_pitch_emb"] = self.avr_energy_emb(inputs.averages["pitch"])

        return conditions

    def _get_style(
        self, inputs: VocoderForwardInput, global_step: int
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        if self.style_feat_type == "linear_spec":
            source, source_lengths = inputs.linear_spectrogram, inputs.spectrogram_lengths
        elif self.style_feat_type == "mel_spec":
            source, source_lengths = inputs.mel_spectrogram, inputs.spectrogram_lengths
        elif self.style_feat_type == "ssl_feat":
            source, source_lengths = inputs.ssl_feat, inputs.ssl_feat_lengths
        elif self.style_feat_type == "speaker_emb":
            source, source_lengths = inputs.speaker_emb, None
        elif self.style_feat_type == "style_emb":
            source, source_lengths = inputs.additional_inputs["style_emb"], None
        else:
            raise NotImplementedError

        if source_lengths is not None:
            source_mask = get_mask_from_lengths(source_lengths)
        else:
            source_mask = None

        style_emb = None
        style_losses = {}

        if self.style_enc is not None:
            style_emb, style_content, style_losses = self.style_enc(
                source, source_mask, inputs=inputs
            )

            if self.style_enc.params.use_gmvae:
                for name, val in style_losses.items():
                    if "kl_loss" in name:
                        val = self.vae_scheduler(global_step, val, name)
                        val = {
                            k: v for k, v in val.items() if not k.startswith("constant")
                        }
                        style_losses.update(val)

        return style_emb, style_losses

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        losses = {}

        x, x_lens = self._get_input_feat(inputs)
        conditions = self._get_conditions(inputs)
        conditions["style_emb"], style_losses = self._get_style(
            inputs, inputs.global_step
        )

        inputs.additional_inputs.update(conditions)
        losses.update(style_losses)

        if self.vq_enc is not None:
            vq_input = ComponentInput(
                content=x, content_lengths=x_lens, model_inputs=inputs
            )
            vq_output = self.vq_enc(vq_input)
            x = vq_output.content

            losses.update(
                {
                    f"codes_{k}": v
                    for k, v in vq_output.additional_losses.items()
                    if not k.startswith("constant")
                }
            )
        else:
            vq_output = ComponentInput.empty()

        if self.energy_predictor is not None:
            e_output, e_content, e_losses = self.energy_predictor(
                x=x,
                x_lengths=x_lens,
                model_inputs=inputs,
                target=inputs.energy,
                name="energy",
            )
            losses.update(e_losses)

        if self.pitch_predictor is not None:
            p_output, p_content, p_losses = self.pitch_predictor(
                x=x,
                x_lengths=x_lens,
                model_inputs=inputs,
                target=inputs.pitch,
                name="pitch",
            )
            losses.update(p_losses)

        if self.range_predictor is not None:
            re = inputs.ranges["energy"]
            rp = inputs.ranges["pitch"]
            target_ranges = torch.stack([re, rp], dim=1)
            feat = torch.cat(
                [
                    conditions["speaker_emb"],
                    conditions["style_emb"],
                ],
                dim=-1,
            )
            ranges = F.relu(self.range_predictor(feat)).reshape(-1, 2, 3)
            losses.update({"range_loss": 0.001 * F.mse_loss(ranges, target_ranges)})

            inputs.energy = inputs.energy * re[:, 2:3] + re[:, 0:1]
            inputs.pitch = inputs.pitch * rp[:, 2:3] + rp[:, 0:1]

        enc_input = ComponentInput(content=x, content_lengths=x_lens, model_inputs=inputs)
        enc_output = self.encoder(enc_input)
        x = enc_output.content

        if self.addm is not None:
            vq_output.additional_content.update(conditions)
            vq_output.additional_losses = {}
            addm_out = self.addm(vq_output)
            losses.update(
                {k: v for k, v in addm_out.additional_losses.items() if "constant" in k}
            )

        if self.mel_predictor is not None:
            enc_input = ComponentInput(
                content=x, content_lengths=x_lens, model_inputs=inputs
            )
            mel_predict = self.mel_predictor(enc_input).content[0]
            losses["auxiliary_mel_loss"] = 0.1 * F.l1_loss(
                mel_predict, inputs.mel_spectrogram
            )

        chunk = []
        for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
            chunk.append(x[i, a:b, :])

        output = torch.stack(chunk)
        return output.transpose(1, -1), losses
