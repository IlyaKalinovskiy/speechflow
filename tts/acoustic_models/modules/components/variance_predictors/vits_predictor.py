import enum
import random
import typing as tp

import torch

from torch import nn

from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.tensor_utils import (
    apply_mask,
    get_lengths_from_durations,
    get_lengths_from_mask,
    get_mask_from_lengths,
)
from tts.acoustic_models.modules.common.length_regulators import SoftLengthRegulator
from tts.acoustic_models.modules.common.vits.normalizing_flow import (
    ResidualCouplingBlock as VITS2RCBlock,
)
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, ComponentOutput
from tts.acoustic_models.modules.params import VariancePredictorParams

RESYNT = False

__all__ = ["VITSPredictor", "VITSPredictorParams"]


class VITSMode(enum.Enum):
    text = 0
    audio = 1
    dual = 2


class VITSPredictorParams(VariancePredictorParams):
    mode: tp.Union[str, VITSMode] = VITSMode.text
    content_encoder_type: tp.Union[str, tp.Tuple[str, str]] = "FFTEncoder"
    content_encoder_params: dict = None  # type: ignore
    audio_feat: str = "linear_spectrogram"
    audio_feat_dim: int = 513
    audio_feat_proj_dim: int = 513
    audio_encoder_type: str = "FFTEncoder"
    audio_encoder_params: dict = None  # type: ignore
    flow_type: str = "VITS2RCBlock"  # VITS2RCBlock
    flow_n_flows: int = 4
    flow_n_layers: int = 4
    flow_n_sqz: int = 2
    flow_condition: tp.Optional[tp.Tuple[str, ...]] = None
    flow_condition_dim: int = 0
    c_kl_content: float = 2.0
    c_kl_audio: float = 0.05
    init_from_checkpoint: str = None

    def model_post_init(self, __context: tp.Any):
        if isinstance(self.mode, str):
            self.mode = eval(f"VITSMode.{self.mode}")
        if self.content_encoder_params is None:
            self.content_encoder_params = {}
        if self.audio_encoder_params is None:
            self.audio_encoder_params = {}
        if isinstance(self.content_encoder_type, str):
            self.content_encoder_type = (
                self.content_encoder_type,
                self.content_encoder_type,
            )


class VITSPredictor(Component):
    params: VITSPredictorParams

    def __init__(
        self, params: VITSPredictorParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)

        if self.params.audio_feat_dim != self.params.audio_feat_proj_dim:
            self.audio_feat_proj = nn.Linear(
                self.params.audio_feat_dim, self.params.audio_feat_proj_dim
            )
        else:
            self.audio_feat_proj = None

        (
            self.content_encoders,
            self.m_p_proj,
            self.log_p_proj,
        ) = self._init_content_encoders(params, input_dim)

        self.audio_encoder, self.m_q_proj, self.log_q_proj = self._init_audio_encoder(
            params
        )

        self.flow = self._init_flow(params)

        self.lr = SoftLengthRegulator()

        if params.init_from_checkpoint is not None:
            checkpoint = ExperimentSaver.load_checkpoint(params.init_from_checkpoint)
            state_dict = checkpoint["state_dict"]

            flow_sd = {}
            audio_encoder_sd = {}
            m_q_proj_sd = {}
            log_q_proj_sd = {}
            for k, v in state_dict.items():
                if "flow" in k:
                    k = k.replace("feature_extractor.vits_feat.flow.", "")
                    flow_sd[k] = v
                if "audio_encoder" in k:
                    k = k.replace("feature_extractor.vits_feat.audio_encoder.", "")
                    audio_encoder_sd[k] = v
                if "m_q_proj" in k:
                    k = k.replace("feature_extractor.vits_feat.m_q_proj.", "")
                    m_q_proj_sd[k] = v
                if "log_q_proj" in k:
                    k = k.replace("feature_extractor.vits_feat.log_q_proj.", "")
                    log_q_proj_sd[k] = v

            params.max_output_length = audio_encoder_sd["position_enc"].shape[1] // 2
            self.audio_encoder, self.m_q_proj, self.log_q_proj = self._init_audio_encoder(
                params
            )

            self.flow.load_state_dict(flow_sd)
            self.audio_encoder.load_state_dict(audio_encoder_sd)
            self.m_q_proj.load_state_dict(m_q_proj_sd)
            self.log_q_proj.load_state_dict(log_q_proj_sd)

    @staticmethod
    def _init_content_encoders(params, input_dim):
        from tts.acoustic_models.modules import TTS_ENCODERS

        if params.mode == VITSMode.dual:
            assert len(input_dim) == 2

        if not isinstance(input_dim, list):
            input_dim = [input_dim]

        encoders = torch.nn.ModuleList()
        m_c_proj = torch.nn.ModuleList()
        log_c_proj = torch.nn.ModuleList()

        if params.mode == VITSMode.dual:
            max_length = [params.max_output_length, params.max_output_length]
        else:
            max_length = [params.max_output_length]

        for i in range(2 if params.mode == VITSMode.dual else 1):
            encoder_cls, encoder_params_cls = TTS_ENCODERS[params.content_encoder_type[i]]

            params.content_encoder_params.update(
                {
                    "encoder_num_layers": params.vp_num_layers,
                    "encoder_inner_dim": params.vp_inner_dim,
                    "encoder_output_dim": params.vp_latent_dim,
                    "max_input_length": max_length[i],
                }
            )

            encoder_params = encoder_params_cls.init_from_parent_params(
                params,
                params.content_encoder_params,
                False,
            )
            encoder = encoder_cls(encoder_params, input_dim[i])
            encoders.append(encoder)

            m_c_proj.append(nn.Linear(encoder.output_dim, params.vp_latent_dim))
            log_c_proj.append(nn.Linear(encoder.output_dim, params.vp_latent_dim))

        if len(m_c_proj) == 1:
            m_c_proj = m_c_proj[0]
            log_c_proj = log_c_proj[0]

        return encoders, m_c_proj, log_c_proj

    @staticmethod
    def _init_audio_encoder(params):
        from tts.acoustic_models.modules import TTS_ENCODERS

        encoder_cls, encoder_params_cls = TTS_ENCODERS[params.audio_encoder_type]
        params.audio_encoder_params.update(
            {
                "encoder_num_layers": params.vp_num_layers,
                "encoder_inner_dim": params.vp_inner_dim,
                "encoder_output_dim": params.vp_latent_dim,
                "max_input_length": params.max_output_length,
            }
        )

        encoder_params = encoder_params_cls.init_from_parent_params(
            params,
            params.audio_encoder_params,
            False,
        )
        encoder = encoder_cls(encoder_params, params.audio_feat_proj_dim)

        m_proj = nn.Linear(encoder.output_dim, params.vp_latent_dim)
        log_proj = nn.Linear(encoder.output_dim, params.vp_latent_dim)
        return encoder, m_proj, log_proj

    def _init_flow(self, params):
        if self.params.flow_type in ["ResidualCouplingBlock", "VITS2RCBlock"]:
            flow_cls = VITS2RCBlock
        else:
            raise NotImplementedError(f"'{self.params.flow_type}' not implemented.")

        return flow_cls(
            params.vp_latent_dim,
            self.params.vp_inner_dim,
            kernel_size=5,
            dilation_rate=1,
            n_layers=self.params.flow_n_layers,
            n_flows=self.params.flow_n_flows,
            gin_channels=params.flow_condition_dim,
        )

    @property
    def output_dim(self):
        return self.params.vp_inner_dim

    @staticmethod
    def _kl_loss(
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ):
        """z_p, logs_q: [b, h, t_t] m_p, logs_p: [b, h, t_t]"""
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(apply_mask(kl, z_mask))
        return kl / torch.sum(z_mask).clamp(min=1)

    @staticmethod
    def _kl_loss_normal(
        m_q: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ):
        """z_p, logs_q: [b, h, t_t] m_p, logs_p: [b, h, t_t]"""
        m_q = m_q.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += (
            0.5 * (torch.exp(2.0 * logs_q) + (m_q - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        )
        kl = torch.sum(apply_mask(kl, z_mask))
        return kl / torch.sum(z_mask).clamp(min=1)

    def _get_target_feat(self, inputs):
        if hasattr(inputs, self.params.audio_feat):
            return getattr(inputs, self.params.audio_feat)
        elif hasattr(inputs, "additional_inputs"):
            return inputs.additional_inputs.get(self.params.audio_feat)
        else:
            raise NotImplementedError

    @staticmethod
    def _get_durations(inputs):
        durations = inputs.durations
        if durations is None:
            for key in inputs.additional_inputs.keys():
                if key.startswith("dura") and key.endswith("postprocessed"):
                    durations = inputs.additional_inputs.get(key)

        if durations is not None:
            durations = durations.squeeze(1).squeeze(-1)

        return durations

    @staticmethod
    def _set_output_lengths(inputs, durations):
        if durations is not None:
            new_lengths = get_lengths_from_durations(durations)
        else:
            new_lengths = None

        if inputs.output_lengths is None:
            inputs.output_lengths = new_lengths

    def _evaluate_content_encoder(self, x, x_mask, y_mask, durations, model_inputs):
        add_content = {}
        add_losses = {}

        if not isinstance(x, list):
            x, x_mask = [x], [x_mask]

        m_and_logs = []
        for i, enc in enumerate(self.content_encoders):
            if x[i] is None:
                m_and_logs.append((None, None))
                continue

            x = x[i]
            x_lens = get_lengths_from_mask(x_mask[i])

            t = ComponentInput(
                content=x,
                content_lengths=x_lens,
                embeddings={},
                model_inputs=model_inputs,
            )
            r: ComponentOutput = enc(t)

            if self.params.mode == VITSMode.dual:
                m_p_proj = self.m_p_proj[i]
                log_p_proj = self.log_p_proj[i]
            else:
                m_p_proj = self.m_p_proj
                log_p_proj = self.log_p_proj

            if r.content.shape[1] != y_mask.shape[1]:
                m_p = apply_mask(m_p_proj(r.content), x_mask[i])
                logs_p = apply_mask(log_p_proj(r.content), x_mask[i])
            else:
                m_p = apply_mask(m_p_proj(r.content), y_mask)
                logs_p = apply_mask(log_p_proj(r.content), y_mask)

            if self.params.mode == VITSMode.text:
                if r.content.shape[1] != y_mask.shape[1]:
                    m_p, _ = self.lr(m_p, durations, max_len=y_mask.shape[1])
                    logs_p, _ = self.lr(logs_p, durations, max_len=y_mask.shape[1])

            m_and_logs.append((m_p, logs_p))

            add_content.update(r.additional_content)
            add_content[f"vits_latent_{i}"] = r.content
            add_losses.update(r.additional_losses)

        return m_and_logs, add_content, add_losses

    def _evaluate_audio_encoder(self, y, y_mask, inputs):
        t = ComponentInput(
            content=y,
            content_lengths=inputs.output_lengths,
            embeddings={},
            model_inputs=inputs,
        )
        r: ComponentOutput = self.audio_encoder(t)

        latent = r.content
        add_content = r.additional_content
        add_losses = r.additional_losses
        return apply_mask(latent, y_mask), add_content, add_losses

    def _get_p(self):
        if isinstance(self.input_dim, int):
            return 0

        if len(self.input_dim) == 1 or RESYNT:
            return 0

        if len(self.input_dim) == 2:
            if self.params.mode == VITSMode.text or self.params.mode == VITSMode.audio:
                return 0
            else:
                if self.training and random.random() < 0.5:
                    return 0
                else:
                    return 1

    def _evaluate_vits(self, z_p, m_p, logs_p, z_q, m_q, logs_q, g, y_mask):
        z_p, m_p, logs_p = (
            z_p.transpose(1, -1),
            m_p.transpose(1, -1),
            logs_p.transpose(1, -1),
        )
        z_q, m_q, logs_q = (
            z_q.transpose(1, -1),
            m_q.transpose(1, -1),
            logs_q.transpose(1, -1),
        )
        y_mask = y_mask.unsqueeze(1)

        z_q_content, m_q_content, logs_q_content = self.flow(
            z_q, y_mask, g=g, m=m_q, logs=logs_q
        )
        z_p_audio, m_p_audio, logs_p_audio = self.flow(
            z_p, y_mask, g=g, reverse=True, m=m_p, logs=logs_p
        )

        loss_kl_content = (
            self._kl_loss(z_q_content, logs_q_content, m_p, logs_p, y_mask)
            * self.params.c_kl_content
        )
        loss_kl_audio = (
            self._kl_loss_normal(m_p_audio, logs_p_audio, m_q, logs_q, y_mask)
            * self.params.c_kl_audio
        )
        return {
            "vits_kl_loss_content": loss_kl_content,
            "vits_kl_loss_audio": loss_kl_audio,
        }

    def forward_step(self, x, x_mask, **kwargs):
        content = {}
        losses = {}

        inputs = kwargs.get("model_inputs")

        y = self._get_target_feat(inputs)
        if y is not None and self.audio_feat_proj is not None:
            y = self.audio_feat_proj(y)

        durations = self._get_durations(inputs)
        if not RESYNT and not self.training:
            self._set_output_lengths(inputs, durations)

        y_mask = get_mask_from_lengths(inputs.output_lengths)

        g = self.get_condition(inputs, self.params.flow_condition)
        if g is not None:
            g = g.transpose(1, -1)

        if "vits_p" not in inputs.additional_inputs:
            m_and_logs, content_p, losses_p = self._evaluate_content_encoder(
                x, x_mask, y_mask, durations, inputs
            )
            content.update(content_p)
            losses.update(losses_p)

            p_index = self._get_p()
            m_p, logs_p = m_and_logs[p_index]
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)

            if self.training and self.params.mode == VITSMode.dual:
                p = torch.distributions.normal.Normal(
                    m_and_logs[0][0], torch.exp(m_and_logs[0][1])
                )
                q = torch.distributions.normal.Normal(
                    m_and_logs[1][0], torch.exp(m_and_logs[1][1])
                )
                losses["vits_dual_loss"] = torch.distributions.kl_divergence(p, q).sum()
        else:
            z_p, m_p, logs_p = inputs.additional_inputs["vits_p"]

        if self.training and y is not None:
            if "vits_q" not in inputs.additional_inputs:
                latent_q, content_q, losses_q = self._evaluate_audio_encoder(
                    y, y_mask, inputs
                )
                content.update(content_q)
                losses.update(losses_q)

                m_q = self.m_q_proj(latent_q)
                logs_q = self.log_q_proj(latent_q)
                z_q = apply_mask(m_q + torch.randn_like(m_q) * torch.exp(logs_q), y_mask)

                content["vits_q"] = (z_q, m_q, logs_q)
            else:
                z_q, m_q, logs_q = inputs.additional_inputs["vits_q"]

            vits_losses = self._evaluate_vits(
                z_p, m_p, logs_p, z_q, m_q, logs_q, g, y_mask
            )
            losses.update(vits_losses)

            return z_q, content, losses
        else:
            _z_p, _m_p, _logs_p = (
                z_p.transpose(1, -1),
                m_p.transpose(1, -1),
                logs_p.transpose(1, -1),
            )
            _z_q, _m_q, _logs_q = self.flow(
                _z_p, y_mask.unsqueeze(1), g=g, reverse=True, m=_m_p, logs=_logs_p
            )
            z_q, m_q, logs_q = (
                _z_q.transpose(1, -1),
                _m_q.transpose(1, -1),
                _logs_q.transpose(1, -1),
            )
            content["vits_q"] = (z_q, m_q, logs_q)
            return z_q, content, {}
