import math
import typing as tp

import torch

from tts.acoustic_models.modules.common.length_regulators import SoftLengthRegulator
from tts.acoustic_models.modules.embedding_calculator import EmbeddingCalculator
from tts.acoustic_models.modules.params import EmbeddingParams
from tts.forced_alignment.data_types import AlignerForwardInput, AlignerForwardOutput
from tts.forced_alignment.model.blocks import (
    AlignmentEncoder,
    FlowSpecDecoder,
    TextEncoder,
)
from tts.forced_alignment.model.utils import (
    binarize_attention,
    generate_path,
    maximum_path,
    sequence_mask,
)

__all__ = ["GlowTTS", "GlowTTSParams"]


class GlowTTSParams(EmbeddingParams):
    """GlowTTS model parameters."""

    flow_type: str = "GlowTTS"  # GlowTTS

    audio_feat: str = "mel"  # mel, ssl
    audio_feat_size: int = 80

    encoder_embedding_dim: int = 128

    inner_channels_enc: int = 192
    inner_channels_dec: int = 192

    filter_channels: int = 768
    filter_channels_dp: int = 256

    kernel_size_enc: int = 3
    kernel_size_dec: int = 5

    n_layers_enc: int = 6
    n_heads_enc: int = 2
    n_blocks_dec: int = 12
    n_layers_dec: int = 4

    window_size: int = 4
    n_split: int = 4
    n_sqz: int = 2
    dilation_rate: int = 1
    p_dropout: float = 0.1

    use_alignment_encoder: bool = False
    alignment_encoder_n_att_channels: int = 128
    alignment_encoder_temperature: float = 0.0005
    alignment_encoder_dist_type: str = "l2"

    use_mas_correction: bool = False
    frames_per_sec: float = 172  # 22050 / 128
    max_phoneme_duration: float = 0.15

    def model_post_init(self, __context: tp.Any):
        super().model_post_init(__context)


class GlowTTS(EmbeddingCalculator):
    params: GlowTTSParams

    def __init__(self, params: tp.Union[GlowTTSParams, dict], strict_init: bool = True):
        super().__init__(GlowTTSParams.create(params, strict_init))
        params = self.params

        self.n_split = params.n_split
        self.n_sqz = params.n_sqz
        self.out_channels = params.audio_feat_size

        self.use_speaker_emb = (
            params.use_onehot_speaker_emb
            or params.use_learnable_speaker_emb
            or params.use_dnn_speaker_emb
            or params.use_mean_dnn_speaker_emb
        )
        self.speaker_emb_dim = (
            params.speaker_emb_dim if self.use_speaker_emb else None  # type: ignore
        )

        self.encoder = TextEncoder(
            params.n_symbols,
            params.n_langs,
            params.n_symbols_per_token,
            params.encoder_embedding_dim,
            self.out_channels,
            params.inner_channels_enc,
            params.filter_channels,
            params.filter_channels_dp,
            params.n_heads_enc,
            params.n_layers_enc,
            params.kernel_size_enc,
            params.p_dropout,
            window_size=params.window_size,
            speaker_emb_dim=self.speaker_emb_dim,
            prenet=True,
        )

        if params.flow_type == "GlowTTS":
            self.decoder = FlowSpecDecoder(
                self.out_channels,
                params.inner_channels_dec,
                params.kernel_size_dec,
                params.dilation_rate,
                params.n_blocks_dec,
                params.n_layers_dec,
                p_dropout=params.p_dropout,
                n_split=self.n_split,
                n_sqz=self.n_sqz,
                speaker_emb_dim=256,
            )
        else:
            raise NotImplementedError(f"'{params.flow_type}' not implemented.")

        self.lang_emb = torch.nn.Embedding(params.n_langs, self.speaker_emb_dim)
        self.cond_proj = torch.nn.Linear(self.speaker_emb_dim * 2 + 4, 256)

        proj_dim = self.out_channels
        self.length_regulator = SoftLengthRegulator()
        self.mel_proj = torch.nn.Sequential(
            torch.nn.Linear(proj_dim, proj_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim * 2, self.out_channels),
        )

        if params.use_alignment_encoder:
            self.alignment_encoder = AlignmentEncoder(
                n_mel_channels=self.out_channels,
                n_text_channels=params.encoder_embedding_dim * 4,
                n_att_channels=params.alignment_encoder_n_att_channels,
                temperature=params.alignment_encoder_temperature,
                dist_type=params.alignment_encoder_dist_type,
            )
        else:
            self.alignment_encoder = None

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (
                torch.div(y_max_length, self.n_sqz, rounding_mode="trunc") * self.n_sqz
            )
            y = y[:, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.n_sqz, rounding_mode="trunc") * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()

    def mas(self, x_m, x_logs, z, attn_mask, inputs, mas_correction):
        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp2 = torch.matmul(
                x_s_sq_r.transpose(1, 2), -0.5 * (z**2)
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul(
                (x_m * x_s_sq_r).transpose(1, 2), z
            )  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m**2) * x_s_sq_r, [1]).unsqueeze(
                -1
            )  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            if mas_correction or self.params.use_mas_correction:
                sil_mask = inputs.ling_feat.sil_mask.cpu().numpy()
                spectral_flatness = inputs.spectral_flatness.cpu().numpy()
                max_frames_per_phoneme = int(
                    self.params.frames_per_sec * self.params.max_phoneme_duration
                )
            else:
                sil_mask = spectral_flatness = max_frames_per_phoneme = None

            attn = maximum_path(
                logp,
                attn_mask.squeeze(1),
                sil_mask=sil_mask,
                spectral_flatness=spectral_flatness,
                max_frames_per_phoneme=max_frames_per_phoneme,
            )

        return attn.unsqueeze(1).detach()

    def calculate_losses(
        self, y, z, attn, x_m, x_logs, logw, logdet, x_mask, x_lengths, y_lengths
    ):
        # [b, t', t], [b, t, d] -> [b, d, t']
        y_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
        ).transpose(1, 2)
        # [b, t', t], [b, t, d] -> [b, d, t']
        y_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
        ).transpose(1, 2)
        logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

        l_mle = 0.5 * math.log(2 * math.pi) + (
            torch.sum(y_logs)
            + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2)
            - torch.sum(logdet)
        ) / (
            torch.sum(torch.div(y_lengths, self.n_sqz, rounding_mode="trunc"))
            * self.n_sqz
            * self.out_channels
        )
        l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        return l_mle, l_length

    def forward(self, inputs: AlignerForwardInput, mas_correction: bool = False) -> AlignerForwardOutput:  # type: ignore
        x = inputs.transcription
        lang_id = inputs.lang_id
        text_lengths = inputs.input_lengths

        if self.params.audio_feat == "mel":
            y = inputs.spectrogram
        elif self.params.audio_feat == "ssl":
            y = inputs.ssl_feat
        else:
            raise ValueError(f"{self.params.audio_feat} is not implemented.")

        y = y.transpose(1, 2)
        output_lengths = inputs.output_lengths

        lang_emb = self.lang_emb(lang_id)
        speaker_emb = self.get_speaker_embedding(inputs)  # type: ignore
        ling_feat_emb = self.get_ling_feat(inputs)  # type: ignore

        x_lengths, y_lengths = text_lengths.data, output_lengths.data
        x, x_m, x_logs, logw, x_mask = self.encoder(
            x,
            lang_emb,
            ling_feat_emb,
            x_lengths,
            g=speaker_emb,
            sil_mask=inputs.ling_feat.sil_mask,
        )

        y_max_length = y.size(2)

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        g = self.cond_proj(
            torch.cat([speaker_emb, lang_emb, inputs.speech_quality_emb], dim=1)
        )
        z, logdet = self.decoder(y, y_mask, g=g.unsqueeze(-1))

        attn = self.mas(x_m, x_logs, z, attn_mask, inputs, mas_correction)

        l_mle, l_length = self.calculate_losses(
            y, z, attn, x_m, x_logs, logw, logdet, x_mask, x_lengths, y_lengths
        )

        if self.alignment_encoder is not None:
            attn_soft, attn_logprob = self.alignment_encoder(
                queries=z,
                keys=x,
                mask=(~x_mask).transpose(2, 1),
                attn_prior=attn.squeeze(1).transpose(1, 2),
            )

            try:
                attn_hard = binarize_attention(
                    attn_soft, inputs.input_lengths, inputs.output_lengths
                )
                aligning_path = attn_hard.squeeze(1)
            except:
                aligning_path = attn.squeeze(1).transpose(1, 2)

            additional_content = {
                "attn_soft": attn_soft,
                "attn_logprob": attn_logprob,
            }
        else:
            aligning_path = attn.squeeze(1).transpose(1, 2)
            additional_content = None

        output = AlignerForwardOutput(
            aligning_path=aligning_path,
            mle_loss=l_mle,
            duration_loss=l_length,
            additional_content=additional_content,
        )
        return output

    @torch.no_grad()
    def generate(self, inputs: AlignerForwardInput):  # type: ignore
        assert inputs.ling_feat
        x = inputs.transcription
        text_lengths = inputs.input_lengths

        if self.params.audio_feat == "mel":
            y = inputs.spectrogram
        elif self.params.audio_feat == "ssl":
            y = inputs.ssl_feat
        else:
            raise ValueError(f"{self.params.audio_feat} is not implemented.")

        y = y.transpose(1, 2)
        output_lengths = inputs.output_lengths

        speaker_emb = self.get_speaker_embedding(inputs)  # type: ignore
        ling_feat_emb = self.get_ling_feat(inputs)  # type: ignore

        x_lengths, y_lengths = text_lengths.data, output_lengths.data
        x_m, x_logs, logw, x_mask = self.encoder(
            x, ling_feat_emb, x_lengths, g=speaker_emb
        )

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        z_m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
        ).transpose(1, 2)
        z_logs = torch.matmul(
            attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
        ).transpose(1, 2)

        z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m)) * z_mask
        y, logdet = self.decoder(z, z_mask, reverse=True, g=speaker_emb)

        output = AlignerForwardOutput(
            spectrogram=y.transpose(1, 2),
            aligning_path=attn.squeeze(1).transpose(1, 2),
            output_mask=torch.LongTensor([[y.shape[2]]]),
        )
        return output
