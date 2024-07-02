import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
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
    add_lm_feat: bool = False
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
            _enc_params.encoder_output_dim = 1
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

        if params.add_lm_feat:
            self.lm_proj = nn.Linear(params.lm_feat_dim, input_dim, bias=False)

        self.lr = SoftLengthRegulator()
        self.hard_lr = SoftLengthRegulator(hard=True)

    @property
    def output_dim(self):
        return 1

    def forward_step(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor,
        model_inputs: MODEL_INPUT_TYPE,
        **kwargs,
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        t_target = kwargs.get("target")
        losses = {}

        w_lengths = model_inputs.additional_inputs.get("word_lengths")
        w_inv_dura = model_inputs.additional_inputs.get("word_invert_lengths")

        x_by_words, _ = self.lr(x.detach(), w_inv_dura, w_lengths.shape[1])

        if self.params.add_lm_feat:
            x_by_words = x_by_words + self.lm_proj(model_inputs.lm_feat)

        w_predict, w_ctx = self.word_encoder.process_content(
            x_by_words, model_inputs.num_words, model_inputs
        )

        w_ctx, _ = self.hard_lr(w_ctx, w_lengths, x.shape[1])

        x_by_tokens = self.token_proj(torch.cat([x, w_ctx], dim=2))
        t_predict, t_ctx = self.token_encoder.process_content(
            x_by_tokens, model_inputs.token_lengths, model_inputs
        )
        t_dura = F.relu(t_predict).squeeze(-1)

        if self.training:
            t_dura = apply_mask(t_dura, get_mask_from_lengths(x_length))

            if self.params.add_noise:
                t_target = (t_target + 0.01 * torch.randn_like(t_target)).clip(min=0.1)

            if model_inputs.global_step % self.params.every_iter == 0:
                losses["dp_loss_by_tokens"] = F.mse_loss(t_dura, t_target)

            durations = t_target
        else:
            durations = t_dura

        return (
            durations,
            {
                "dp_context": t_ctx,
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
            in_channels=self.token_encoder.params.encoder_inner_dim
        )

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        dur_predict, dur_content, dur_losses = super().forward_step(
            x, x_lengths, model_inputs, **kwargs
        )

        if self.training:
            dur_real = model_inputs.durations.unsqueeze(1)
            dur_fake = dur_content["dp_predict"].unsqueeze(1)
            context = dur_content["dp_context"]
            mask = get_mask_from_lengths(x_lengths)

            disc_losses = self.disc.calculate_loss(
                context.transpose(2, 1),
                mask.unsqueeze(1),
                dur_real,
                dur_fake,
                model_inputs.global_step,
            )
            dur_losses.update({f"dp_{k}": v for k, v in disc_losses.items()})

        return dur_predict, dur_content, dur_losses
