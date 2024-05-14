import typing as tp

import torch

from pydantic import Field
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
    add_noise: bool = True
    every_iter: int = 2


class TokenLevelDP(Component):
    params: TokenLevelDPParams

    def __init__(
        self, params: TokenLevelDPParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import PARALLEL_ENCODERS

        def _init_encoder(_enc_cls, _enc_params_cls, _encoder_params):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = 2
            return _enc_cls(_enc_params, input_dim)

        enc_cls, enc_params_cls = PARALLEL_ENCODERS[params.word_encoder_type]
        self.word_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.word_encoder_params
        )

        enc_cls, enc_params_cls = PARALLEL_ENCODERS[params.token_encoder_type]
        self.token_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.token_encoder_params
        )
        self.token_proj = torch.nn.Linear(
            input_dim + params.vp_inner_dim + 1, input_dim, bias=False
        )

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

        x_by_words, _ = self.lr(x, w_inv_dura, w_lengths.shape[1])
        w_predict, w_ctx = self.token_encoder.process_content(
            x_by_words, m_inputs.num_words, m_inputs
        )
        w_predict = F.relu(w_predict)

        w_lens = torch.expm1(w_predict[..., 1]).unsqueeze(-1)
        if self.training:
            w_lens = w_target.unsqueeze(-1)

        w_ctx, _ = self.hard_lr(w_ctx, w_lengths, x.shape[1])
        w_lens, _ = self.hard_lr(w_lens, w_lengths, x.shape[1])

        x_by_tokens = self.token_proj(torch.cat([x, w_ctx, w_lens], dim=2))
        t_predict, t_ctx = self.token_encoder.process_content(
            x_by_tokens, m_inputs.token_lengths, m_inputs
        )
        t_predict = F.relu(t_predict)

        durations_predict = torch.expm1(t_predict[..., 0])
        # durations_predict = p_proj[..., 1] * w_len.squeeze(-1)

        if self.training:
            d = m_inputs.durations
            if self.params.add_noise:
                w_target = (w_target + torch.randn_like(w_target)).clip(min=0.1)
                d = (d + 0.1 * torch.randn_like(d)).clip(min=0.1)

            if m_inputs.global_step % self.params.every_iter == 0:
                losses["dur_token_loss"] = F.l1_loss(w_predict[..., 0], w_target)
                losses["dur_token_log_loss"] = F.l1_loss(
                    w_predict[..., 1], torch.log1p(w_target)
                )
                losses["dur_phoneme_loss"] = F.l1_loss(t_predict[..., 0], torch.log1p(d))
                losses["dur_phoneme_rel_loss"] = F.l1_loss(
                    t_predict[..., 1], d / w_lens.squeeze(-1).clip(min=0.1)
                )

            durations = m_inputs.durations
        else:
            durations = durations_predict

        return (
            durations,
            {
                "dp_context": t_ctx,
                "dp_context_mask": x_mask,
                "dp_predict": durations_predict,
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
