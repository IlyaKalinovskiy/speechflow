import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = [
    "TokenLevelDP",
    "TokenLevelDPParams",
    "TokenLevelDPWithDiscriminator",
    "TokenLevelDPWithDiscriminatorParams",
]


class TokenLevelDPParams(VariancePredictorParams):
    word_encoder_type: str = "RNNEncoder"
    word_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    token_encoder_type: str = "RNNEncoder"
    token_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    use_lm: bool = False
    add_noise: bool = False
    every_iter: int = 2


class TokenLevelDP(Component):
    params: TokenLevelDPParams

    def __init__(
        self, params: TokenLevelDPParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        def _init_encoder(_enc_cls, _enc_params_cls, _encoder_params):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = 2
            return _enc_cls(_enc_params, input_dim)

        enc_cls, enc_params_cls = TTS_ENCODERS[params.word_encoder_type]
        self.word_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.word_encoder_params
        )

        enc_cls, enc_params_cls = TTS_ENCODERS[params.token_encoder_type]
        self.token_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.token_encoder_params
        )
        self.token_proj = nn.Linear(
            input_dim + params.vp_inner_dim, input_dim, bias=False
        )

        if params.use_lm:
            self.lm_proj = nn.Linear(params.lm_feat_dim, input_dim, bias=False)

        self.lr = SoftLengthRegulator()
        self.hard_lr = SoftLengthRegulator(hard=True)

    @property
    def output_dim(self):
        return 1

    def forward_step(
        self, x: torch.Tensor, x_mask: torch.BoolTensor, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        m_inputs = kwargs.get("model_inputs")
        losses = {}

        w_target = m_inputs.additional_inputs.get("word_durations")
        w_lengths = m_inputs.additional_inputs.get("word_lengths")
        w_inv_dura = m_inputs.additional_inputs.get("word_invert_lengths")

        t_target = m_inputs.durations

        x_by_words, _ = self.lr(x.detach(), w_inv_dura, w_lengths.shape[1])

        if self.params.use_lm:
            x_by_words = x_by_words + self.lm_proj(m_inputs.lm_feat)

        w_predict, w_ctx = self.token_encoder.process_content(
            x_by_words, m_inputs.num_words, m_inputs
        )
        w_predict = F.relu(w_predict)

        w_dura = torch.expm1(w_predict[..., 1]).unsqueeze(-1)
        if self.training:
            w_dura = w_target.unsqueeze(-1)

        w_ctx, _ = self.hard_lr(w_ctx, w_lengths, x.shape[1])
        w_dura, _ = self.hard_lr(w_dura, w_lengths, x.shape[1])

        x_by_tokens = self.token_proj(torch.cat([x, w_ctx], dim=2))
        t_predict, t_ctx = self.token_encoder.process_content(
            x_by_tokens, m_inputs.token_lengths, m_inputs
        )
        t_predict = F.relu(t_predict)

        t_dura = torch.expm1(t_predict[..., 0])
        # t_dura = p_proj[..., 1] * w_dura.squeeze(-1)

        if self.training:
            if self.params.add_noise:
                w_target = (w_target + 0.1 * torch.randn_like(w_target)).clip(min=0.1)
                t_target = (t_target + 0.1 * torch.randn_like(t_target)).clip(min=0.1)

            if m_inputs.global_step % self.params.every_iter == 0:
                losses["dur_loss_by_words"] = F.l1_loss(w_predict[..., 0], w_target)
                losses["dur_log_loss_by_words"] = F.l1_loss(
                    w_predict[..., 1], torch.log1p(w_target)
                )
                losses["dur_log_loss_by_tokens"] = F.l1_loss(
                    t_predict[..., 0], torch.log1p(t_target)
                )
                losses["dur_rel_loss_by_tokens"] = F.l1_loss(
                    t_predict[..., 1], t_target / w_dura.squeeze(-1).clip(min=0.01)
                )

            durations = m_inputs.durations
        else:
            durations = t_dura

        return (
            durations.squeeze(-1),
            {
                "dp_context": t_ctx,
                "dp_context_mask": x_mask,
                "dp_predict": t_dura,
            },
            losses,
        )


class TokenLevelDPWithDiscriminatorParams(TokenLevelDPParams):
    pass


class TokenLevelDPWithDiscriminator(TokenLevelDP, Component):
    params: TokenLevelDPWithDiscriminatorParams

    def __init__(
        self,
        params: TokenLevelDPWithDiscriminatorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)
        self.disc = SignalDiscriminator(
            in_channels=params.token_encoder_params["encoder_hidden_dim"]
        )

    def forward_step(self, x, mask, **kwargs):
        dur_predict, dur_content, dur_losses = super().forward_step(x, mask, **kwargs)
        m_inputs = kwargs.get("model_inputs")

        if self.training:
            inputs = kwargs.get("model_inputs")
            dur_real = torch.log1p(inputs.durations).unsqueeze(1)
            dur_fake = torch.log1p(dur_content["dp_predict"]).unsqueeze(1)
            context = dur_content["dp_context"]
            context_mask = dur_content["dp_context_mask"]

            disc_losses = self.disc.calculate_loss(
                context.transpose(2, 1),
                context_mask.unsqueeze(1),
                dur_real,
                dur_fake,
                m_inputs.global_step,
            )
            dur_losses.update({f"dur_{k}": v for k, v in disc_losses.items()})

        return dur_predict, dur_content, dur_losses
