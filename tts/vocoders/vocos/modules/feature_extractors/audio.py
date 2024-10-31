import typing as tp

import torch

from torch import nn
from torch.nn import functional as F

from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.training.base_model import BaseTorchModelParams
from speechflow.training.losses.vae_loss import VAELoss
from speechflow.utils.tensor_utils import (
    apply_mask,
    get_lengths_from_durations,
    get_mask_from_lengths,
)
from tts.acoustic_models.modules import TTS_ENCODERS
from tts.acoustic_models.modules.additional_modules import (
    AdditionalModules,
    AdditionalModulesParams,
)
from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.common.blocks import Regression, VarianceEmbedding
from tts.acoustic_models.modules.components.encoders import (
    SFEncoderWithClassificationAdaptor,
    SFEncoderWithClassificationAdaptorParams,
    VQEncoderWithClassificationAdaptor,
    VQEncoderWithClassificationAdaptorParams,
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
from tts.acoustic_models.modules.params import VarianceParams
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors.base import FeatureExtractor

__all__ = ["AudioFeatures", "AudioFeaturesParams"]


class AudioFeaturesParams(BaseTorchModelParams):
    input_feat_type: tp.Literal[
        "linear_spectrogram", "mel_spectrogram", "ssl_feat"
    ] = "mel_spectrogram"
    input_proj_dim: int = 256
    inner_dim: int = 512
    # embeddings
    n_langs: int = 1
    n_speakers: int = 1
    n_centroids: int = 1024
    lang_emb_dim: int = 32
    speaker_emb_dim: int = 256
    ssl_feat_dim: int = 1024
    style_emb_dim: int = 128
    linear_spectrogram_dim: int = 513
    mel_spectrogram_dim: int = 80
    average_emb_dim: int = 32
    # feat encoder
    feat_encoder_type: str = "RNNEncoder"
    feat_encoder_num_layers: int = 1
    feat_encoder_inner_dim: int = 512
    feat_encoder_use_condition: bool = True
    feat_condition_type: tp.Literal["cat", "adanorm"] = "cat"
    # variance predictor encoder
    vp_encoder_type: str = "RNNEncoder"
    vp_encoder_num_layers: int = 1
    vp_encoder_inner_dim: int = 512
    vp_encoder_use_condition: bool = True
    vp_condition_type: tp.Literal["cat", "adanorm"] = "cat"
    # vq encoder
    vq_encoder_type: str = "RNNEncoder"
    vq_encoder_num_layers: int = 1
    vq_encoder_inner_dim: int = 512
    vq_encoder_use_condition: bool = True
    vq_condition_type: tp.Literal["cat", "adanorm"] = "cat"
    vq_type: tp.Literal["vq", "rvq", "rfsq", "rlfq"] = "rlfq"
    vq_emb_dim: int = 256
    vq_codebook_size: int = 1024
    vq_num_quantizers: int = 1
    # source-filter encoder
    sf_encoder_type: str = "RNNEncoder"
    sf_encoder_num_layers: int = 1
    sf_encoder_inner_dim: int = 512
    sf_encoder_use_condition: bool = True
    sf_condition_type: tp.Literal["cat", "adanorm"] = "cat"
    # style encoder
    style_encoder_type: tp.Literal[
        "SimpleStyle", "StyleSpeech", "StyleTTS2"
    ] = "StyleSpeech"
    style_feat_type: tp.Literal[
        "linear_spectrogram",
        "mel_spectrogram",
        "ssl_feat",
        "speaker_emb",
        "style_emb",
    ] = "mel_spectrogram"
    style_use_gmvae: bool = False
    style_use_fsq: bool = False
    style_gmvae_n_components: int = 16
    # variances
    energy_interval: tp.Tuple[float, float] = (0, 150)
    pitch_interval: tp.Tuple[float, float] = (0, 880)
    average_energy_interval: tp.Tuple[float, float] = (0, 150)
    average_pitch_interval: tp.Tuple[float, float] = (0, 880)
    energy_denormalize: bool = False
    pitch_denormalize: bool = False
    # Multilingual PL-BERT
    plbert_emb_dim: int = 768
    plbert_proj_dim: int = 256
    # flags
    use_lang_emb: bool = False
    use_speaker_emb: bool = False
    use_speech_quality_emb: bool = False
    use_style: bool = False
    use_energy: bool = False
    use_pitch: bool = False
    use_plbert: bool = False
    use_ssl_adjustment: bool = False
    use_averages: bool = False
    use_vq: bool = False
    use_upsample: bool = False
    use_auxiliary_classification_loss: bool = False
    use_auxiliary_linear_spec_loss: bool = False
    use_auxiliary_mel_spec_loss: bool = False
    use_inverse_grad: bool = False


class AudioFeatures(FeatureExtractor):
    params: AudioFeaturesParams

    def __init__(self, params: AudioFeaturesParams):
        super().__init__(params)

        max_input_length = 1024 * 4
        max_output_length = 1024 * 4

        def _get_feat_dim(feat_name: str) -> int:
            if feat_name == "linear_spectrogram":
                return params.linear_spectrogram_dim
            elif feat_name == "mel_spectrogram":
                return params.mel_spectrogram_dim
            elif feat_name == "ssl_feat":
                return params.ssl_feat_dim
            elif params.style_feat_type == "speaker_emb":
                return params.speaker_emb_dim
            elif params.style_feat_type == "style_emb":
                return params.style_emb_dim
            else:
                raise NotImplementedError(f"feat_name '{feat_name}' is not supported")

        self.input_feat_type = params.input_feat_type
        self.style_feat_type = params.style_feat_type

        condition = []
        condition_dim = 0

        if params.use_lang_emb:
            self.lang_embs = nn.Embedding(params.n_langs, params.lang_emb_dim)
            condition.append("lang_emb")
            condition_dim += params.lang_emb_dim
        else:
            self.lang_embs = None

        if params.use_speaker_emb:
            condition.append("speaker_emb")
            condition_dim += params.speaker_emb_dim

        if params.n_centroids > 0:
            self.ssl_embs = nn.Embedding(params.n_centroids + 1, params.input_proj_dim)
        else:
            self.ssl_embs = None

        if params.use_speech_quality_emb:
            condition.append("speech_quality_emb")
            condition_dim += 4

        if params.use_averages and params.use_energy:
            self.avr_energy_emb = VarianceEmbedding(
                interval=params.average_energy_interval, emb_dim=params.average_emb_dim
            )
            condition.append("avr_energy_emb")
            condition_dim += params.average_emb_dim
        else:
            self.avr_energy_emb = None

        if params.use_averages and params.use_pitch:
            self.avr_pitch_emb = VarianceEmbedding(
                interval=params.average_pitch_interval,
                emb_dim=params.average_emb_dim,
                log_scale=True,
            )
            condition.append("avr_pitch_emb")
            condition_dim += params.average_emb_dim
        else:
            self.avr_pitch_emb = None

        if params.use_style:
            style_params = StyleEncoderParams(
                base_encoder_type=params.style_encoder_type,
                source=None,
                source_dim=_get_feat_dim(params.style_feat_type),
                style_emb_dim=params.style_emb_dim,
                min_spec_len=128,
                max_spec_len=512,
                use_gmvae=params.style_use_gmvae,
                use_fsq=params.style_use_fsq,
                gmvae_n_components=params.style_gmvae_n_components,
            )
            self.style_enc = StyleEncoder(style_params, 0)

            if params.style_use_gmvae:
                self.vae_scheduler = VAELoss(
                    scale=0.00002,
                    every_iter=1,
                    begin_iter=1000,
                    end_anneal_iter=10000,
                )
            else:
                self.vae_scheduler = None

            condition.append("style_emb")
            condition_dim += params.style_emb_dim
        else:
            self.style_enc = None

        if (params.use_energy or params.use_pitch) and (
            params.energy_denormalize or params.pitch_denormalize
        ):
            self.range_predictor = Regression(condition_dim, 3 * 2, activation_fn="ReLU")
        else:
            self.range_predictor = None

        in_dim = params.input_proj_dim
        feat_dim = _get_feat_dim(params.input_feat_type)

        if in_dim != feat_dim:
            self.input_proj = nn.Linear(feat_dim, in_dim)
        else:
            self.input_proj = None

        # ----- init VQ encoder -----

        if params.use_vq:
            vq_encoder_params = {
                "encoder_inner_dim": params.vq_encoder_inner_dim,
                "encoder_num_layers": params.vq_encoder_num_layers,
                "cnn_n_layers": 3,
                "max_input_length": max_input_length,
                "max_output_length": max_output_length,
            }
            if params.vq_encoder_use_condition:
                vq_encoder_params.update(
                    {
                        "condition": condition,
                        "condition_dim": condition_dim,
                        "condition_type": params.vq_condition_type,
                    }
                )

            vq_enc_params = VQEncoderWithClassificationAdaptorParams(
                vq_type=params.vq_type,
                vq_codebook_size=params.vq_codebook_size,
                vq_num_quantizers=params.vq_num_quantizers,
                vq_encoder_type=params.vq_encoder_type,
                vq_encoder_params=vq_encoder_params,
                encoder_output_dim=params.vq_emb_dim,
                tag="vq_encoder",
            )
            self.vq_enc = VQEncoderWithClassificationAdaptor(vq_enc_params, in_dim)
            in_dim = self.vq_enc.output_dim
        else:
            self.vq_enc = None

        # ----- init feat encoder -----

        enc_cls, enc_params_cls = TTS_ENCODERS[params.feat_encoder_type]
        enc_params = enc_params_cls(
            encoder_num_layers=params.feat_encoder_num_layers,
            encoder_inner_dim=params.feat_encoder_inner_dim,
            encoder_output_dim=params.inner_dim,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        if (
            params.feat_encoder_type != "DummyEncoder"
            and params.feat_encoder_use_condition
        ):
            enc_params.condition = tuple(condition)
            enc_params.condition_dim = condition_dim
            enc_params.condition_type = params.feat_condition_type

        self.feat_encoder = enc_cls(enc_params, in_dim)
        in_dim = self.feat_encoder.output_dim

        # ----- init plbert projection -----

        if params.use_plbert:
            self.plbert_proj = nn.Sequential(
                nn.Linear(params.plbert_emb_dim, params.plbert_proj_dim),
                nn.Dropout2d(0.2),
            )
            in_dim += params.plbert_proj_dim
        else:
            self.plbert_proj = None

        # ----- init upsampling -----

        if params.use_upsample:
            self.lr = SoftLengthRegulator(sigma=0.9)
            self.register_buffer("mean_scale_factor", torch.tensor(1.5))
        else:
            self.lr = None

        # ----- init 1d predictors -----

        var_encoder_params = {
            "max_input_length": max_input_length,
            "max_output_length": max_output_length,
        }

        if params.vp_encoder_use_condition:
            var_encoder_params.update(
                {
                    "condition": condition,
                    "condition_dim": condition_dim,
                    "condition_type": params.vp_condition_type,
                }
            )

        if params.use_energy:
            energy_predictor_params = FrameLevelPredictorWithDiscriminatorParams(
                frame_encoder_type=params.vp_encoder_type,
                frame_encoder_params=var_encoder_params,
                activation_fn="ReLU",
                vp_inner_dim=params.vp_encoder_inner_dim,
                vp_num_layers=params.vp_encoder_num_layers,
                vp_output_dim=1,
                var_params=VarianceParams(log_scale=False),
            )
            self.energy_predictor = FrameLevelPredictorWithDiscriminator(
                energy_predictor_params, in_dim
            )
        else:
            self.energy_predictor = None

        if params.use_pitch:
            pitch_predictor_params = FrameLevelPredictorWithDiscriminatorParams(
                frame_encoder_type=params.vp_encoder_type,
                frame_encoder_params=var_encoder_params,
                activation_fn="ReLU",
                vp_inner_channels=params.vp_encoder_inner_dim,
                vp_num_layers=params.vp_encoder_num_layers,
                vp_output_dim=1,
                var_params=VarianceParams(log_scale=True),
                use_ssl_adjustment=params.use_ssl_adjustment,
                ssl_feat_dim=params.ssl_feat_dim,
            )
            self.pitch_predictor = FrameLevelPredictorWithDiscriminator(
                pitch_predictor_params, in_dim
            )
        else:
            self.pitch_predictor = None

        # ----- source-filter encoder -----

        enc_params = SFEncoderWithClassificationAdaptorParams(
            base_encoder_type=params.sf_encoder_type,
            encoder_inner_dim=params.sf_encoder_inner_dim,
            encoder_num_layers=params.sf_encoder_num_layers,
            encoder_output_dim=params.inner_dim,
            var_as_embedding=(True, True),
            var_interval=(params.energy_interval, params.pitch_interval),
            var_log_scale=(False, True),
            max_input_length=max_input_length,
            max_output_length=max_output_length,
        )
        if params.sf_encoder_use_condition:
            enc_params.condition = tuple(condition)
            enc_params.condition_dim = condition_dim
            enc_params.condition_type = params.feat_condition_type

        self.sf_encoder = SFEncoderWithClassificationAdaptor(enc_params, in_dim)

        # ----- init additional modules -----

        if params.use_auxiliary_classification_loss:
            text_proc = TTSTextProcessor(lang="MULTILANG")
            addm_params = AdditionalModulesParams()
            addm_params.alphabet_size = text_proc.alphabet_size
            addm_params.n_symbols_per_token = text_proc.num_symbols_per_phoneme_token
            addm_params.n_speakers = params.n_speakers
            addm_params.speaker_emb_dim = params.speaker_emb_dim

            content_name = self.sf_encoder.name
            content_dim = self.sf_encoder.output_dim
            addm_params.addm_apply_phoneme_classifier = {content_name: content_dim}

            if params.use_vq and params.use_inverse_grad:
                addm_params.addm_apply_inverse_speaker_emb = {
                    "vq_encoder": self.vq_enc.output_dim
                }

            if params.use_style and params.use_inverse_grad:
                addm_params.addm_apply_inverse_speaker_classifier[
                    "style_emb"
                ] = self.style_enc.output_dim

            self.addm = AdditionalModules(addm_params)
        else:
            self.addm = None

        if params.use_auxiliary_linear_spec_loss:
            self.linear_spec_proj = Regression(
                params.inner_dim, params.linear_spectrogram_dim
            )
        else:
            self.linear_spec_proj = None

        if params.use_auxiliary_mel_spec_loss:
            self.mel_spec_proj = Regression(params.inner_dim, params.mel_spectrogram_dim)
        else:
            self.mel_spec_proj = None

    def _get_input_feat(
        self, inputs: VocoderForwardInput
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if self.input_feat_type == "linear_spectrogram":
            return inputs.linear_spectrogram, inputs.spectrogram_lengths
        elif self.input_feat_type == "mel_spectrogram":
            return inputs.spectrogram, inputs.spectrogram_lengths
        elif self.input_feat_type == "ssl_feat":
            return inputs.ssl_feat, inputs.ssl_feat_lengths

    def _get_conditions(self, inputs: VocoderForwardInput) -> tp.Dict[str, tp.Any]:
        conditions = {}

        if self.lang_embs is not None:
            lang_id = inputs.lang_id
            conditions["lang_emb"] = self.lang_embs(lang_id)

        if self.params.use_speaker_emb is not None:
            if inputs.speaker_emb_mean is not None:
                conditions["speaker_emb"] = inputs.speaker_emb_mean
            else:
                conditions["speaker_emb"] = inputs.speaker_emb

        if self.params.use_speech_quality_emb:
            conditions["speech_quality_emb"] = inputs.speech_quality_emb

        if self.avr_energy_emb is not None:
            conditions["avr_energy_emb"] = self.avr_energy_emb(inputs.averages["energy"])

        if self.avr_pitch_emb is not None:
            conditions["avr_pitch_emb"] = self.avr_pitch_emb(inputs.averages["pitch"])

        return conditions

    def _get_style(
        self, inputs: VocoderForwardInput, global_step: int
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:
        if self.style_feat_type == "linear_spectrogram":
            source, source_lengths = inputs.linear_spectrogram, inputs.spectrogram_lengths
        elif self.style_feat_type == "mel_spectrogram":
            source, source_lengths = inputs.spectrogram, inputs.spectrogram_lengths
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
                source, source_mask, model_inputs=inputs
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

    def _cat_plbert_feat(self, x, inputs: VocoderForwardInput):
        plbert_feat = self.plbert_proj(inputs.plbert_feat)

        if x.shape[1] > plbert_feat.shape[1]:
            plbert_feat = F.pad(plbert_feat, (0, 0, 0, x.shape[1] - plbert_feat.shape[1]))
        elif x.shape[1] < plbert_feat.shape[1]:
            plbert_feat = plbert_feat[:, : x.shape[1]]

        return torch.cat([x, plbert_feat], dim=-1)

    def _upsample(self, x, x_lens, inputs: VocoderForwardInput):
        if self.training:
            scale_factor = inputs.spectrogram_lengths / x_lens
            dura = scale_factor.unsqueeze(-1).repeat(1, x.shape[1]).to(x.device)
            dura = apply_mask(dura, get_mask_from_lengths(x_lens, x.shape[1]))
            y, _ = self.lr(x, dura)
            y_lens = inputs.spectrogram_lengths
            self.mean_scale_factor = scale_factor.mean()
        else:
            scale_factor = self.mean_scale_factor.unsqueeze(-1)
            dura = scale_factor.repeat(x.shape[0], x.shape[1]).to(x.device)
            dura = apply_mask(dura, get_mask_from_lengths(x_lens, x.shape[1]))
            y, _ = self.lr(x, dura)

            if inputs.output_lengths is not None:
                y_lens = inputs.output_lengths
            else:
                y_lens = get_lengths_from_durations(dura)

        if inputs.output_lengths is not None:
            if y.shape[1] < inputs.spectrogram.shape[1]:
                y = F.pad(y, (0, 0, 0, inputs.spectrogram.shape[1] - y.shape[1]))
            else:
                y = y[:, : inputs.spectrogram.shape[1], :]

        return y, y_lens

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        losses = {}
        additional_content = {}

        x, x_lens = self._get_input_feat(inputs)
        conditions = self._get_conditions(inputs)

        conditions["style_emb"], style_losses = self._get_style(
            inputs, inputs.global_step
        )
        losses.update(style_losses)
        inputs.additional_inputs.update(conditions)

        if x.dtype == torch.int64:
            x = self.ssl_embs(x.squeeze(-1))
        elif self.input_proj is not None:
            x = self.input_proj(x)

        if self.vq_enc is not None:
            vq_input = ComponentInput(
                content=x, content_lengths=x_lens, model_inputs=inputs
            )
            vq_output = self.vq_enc(vq_input)
            x = vq_output.content
            additional_content.update(vq_output.additional_content)

            losses.update(
                {
                    f"codes_{k}": v
                    for k, v in vq_output.additional_losses.items()
                    if not k.startswith("constant")
                }
            )

        feat_enc_input = ComponentInput(
            content=x, content_lengths=x_lens, model_inputs=inputs
        )
        feat_enc_output = self.feat_encoder(feat_enc_input)
        x = self.feat_encoder.get_content(feat_enc_output)[0]

        if self.params.use_plbert:
            x = self._cat_plbert_feat(x, inputs)

        if self.lr is not None:
            x, x_lens = self._upsample(x, x_lens, inputs)
            feat_enc_output.additional_content["upsample_content"] = x

        if self.energy_predictor is not None:
            e_output, e_content, e_losses = self.energy_predictor(
                x=x.detach(),
                x_lengths=x_lens,
                model_inputs=inputs,
                target=inputs.energy,
                name="energy",
            )
            losses.update(e_losses)
            if not self.training and inputs.energy is None:
                inputs.energy = e_output

        if self.pitch_predictor is not None:
            p_output, p_content, p_losses = self.pitch_predictor(
                x=x.detach(),
                x_lengths=x_lens,
                model_inputs=inputs,
                target=inputs.pitch,
                name="pitch",
            )
            losses.update(p_losses)
            if not self.training and inputs.pitch is None:
                inputs.pitch = p_output

        if self.range_predictor is not None:
            feat = torch.cat(list(conditions.values()), dim=-1).detach()
            ranges_predict = self.range_predictor(feat).reshape(-1, 2, 3)

            if self.training:
                re = inputs.ranges["energy"]
                rp = inputs.ranges["pitch"]
                target_ranges = torch.stack([re, rp], dim=1)
                losses.update(
                    {"range_loss": 0.001 * F.mse_loss(ranges_predict, target_ranges)}
                )
            else:
                if inputs.ranges is None:
                    re = ranges_predict[:, 0]
                    rp = ranges_predict[:, 1]
                else:
                    re = inputs.ranges["energy"]
                    rp = inputs.ranges["pitch"]

            if self.params.energy_denormalize:
                inputs.energy = inputs.energy * re[:, 2:3] + re[:, 0:1]

            if self.params.pitch_denormalize:
                inputs.pitch = inputs.pitch * rp[:, 2:3] + rp[:, 0:1]

        sf_enc_input = ComponentInput(
            content=x, content_lengths=x_lens, model_inputs=inputs
        )
        sf_enc_output = self.sf_encoder(sf_enc_input)
        x = self.sf_encoder.get_content(sf_enc_output)[0]

        if self.training and self.addm is not None:
            sf_enc_output.additional_content.update(conditions)
            sf_enc_output.additional_losses = {}
            addm_out = self.addm(sf_enc_output)
            losses.update(
                {
                    k: v
                    for k, v in addm_out.additional_losses.items()
                    if "constant" not in k
                }
            )

        if self.training and self.linear_spec_proj is not None:
            losses["auxiliary_linear_spec_loss"] = 0.1 * F.mse_loss(
                self.linear_spec_proj(x), inputs.linear_spectrogram
            )

        if self.training and self.mel_spec_proj is not None:
            losses["auxiliary_mel_spec_loss"] = 0.1 * F.mse_loss(
                self.mel_spec_proj(x), inputs.mel_spectrogram
            )

        if "spec_chunk" in inputs.additional_inputs:
            chunk = []
            energy = []
            pitch = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                chunk.append(x[i, a:b, :])
                energy.append(inputs.energy[i, a:b])
                pitch.append(inputs.pitch[i, a:b])

            output = torch.stack(chunk)
            additional_content["energy"] = torch.stack(energy)
            additional_content["pitch"] = torch.stack(pitch)
        else:
            output = x
            additional_content["energy"] = inputs.energy
            additional_content["pitch"] = inputs.pitch

        additional_content["condition_emb"] = torch.cat(list(conditions.values()), dim=-1)
        return output.transpose(1, -1), losses, additional_content
