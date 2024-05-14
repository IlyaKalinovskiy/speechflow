import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import (
    get_lengths_from_durations,
    get_mask_from_lengths,
)
from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = [
    "TokenLevelPredictor",
    "TokenLevelPredictorParams",
    "TokenLevelPredictorWithDiscriminator",
    "TokenLevelPredictorWithDiscriminatorParams",
]


class TokenLevelPredictorParams(VariancePredictorParams):
    word_encoder_type: str = "RNNEncoder"
    word_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    token_encoder_type: str = "RNNEncoder"
    token_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    use_lm: bool = False


class TokenLevelPredictor(Component):
    params: TokenLevelPredictorParams

    def __init__(
        self,
        params: TokenLevelPredictorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import PARALLEL_ENCODERS

        def _init_encoder(_enc_cls, _enc_params_cls, _encoder_params):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = params.vp_output_dim
            return _enc_cls(_enc_params, input_dim)

        enc_cls, enc_params_cls = PARALLEL_ENCODERS[params.word_encoder_type]
        self.word_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.word_encoder_params
        )

        enc_cls, enc_params_cls = PARALLEL_ENCODERS[params.token_encoder_type]
        self.token_proj = nn.Linear(params.vp_inner_dim + input_dim, input_dim)
        self.token_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.token_encoder_params
        )

        if params.use_lm:
            self.lm_proj = nn.Linear(params.lm_feat_dim, input_dim)

        self.lr = SoftLengthRegulator()
        self.hard_lr = SoftLengthRegulator(hard=True)

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_mask, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")
        m_inputs = kwargs.get("model_inputs")

        losses = {}

        word_length = m_inputs.additional_inputs.get("word_lengths")
        word_inv_lengths = m_inputs.additional_inputs.get("word_invert_lengths")

        max_num_tokens = m_inputs.transcription_lengths.max().item()
        max_num_words = m_inputs.num_words.max().item()

        x_by_words, _ = self.lr(x, word_inv_lengths, max_num_words)
        x_by_tokens = x

        wd_mask = get_mask_from_lengths(m_inputs.num_words)
        tk_mask = get_mask_from_lengths(m_inputs.transcription_lengths)

        if self.params.use_lm:
            wd = x_by_words + self.lm_proj(m_inputs.lm_feat)
        else:
            wd = x_by_words

        wd_predict, wd_ctx = self.word_encoder.process_content(
            wd, m_inputs.num_words, m_inputs
        )
        wd_enc, _ = self.hard_lr(wd_ctx, word_length, max_num_tokens)

        tk_proj = self.token_proj(torch.cat([wd_enc, x_by_tokens], dim=2))
        tk_predict, tk_ctx = self.token_encoder.process_content(
            tk_proj, m_inputs.transcription_lengths, m_inputs
        )

        context = tk_ctx
        context_mask = tk_mask
        var_predict = tk_predict.squeeze(-1)

        if self.training:
            target = kwargs.get("target")
            if target.ndim == 2:
                target = target.unsqueeze(-1)

            target_by_words = self.lr(target, word_inv_lengths, max_num_words)[0]
            target_by_tokens = target

            if wd_predict is not None and target_by_words is not None:
                losses[f"{name}_word_loss"] = F.l1_loss(wd_predict, target_by_words)

            if tk_predict is not None and target_by_tokens is not None:
                losses[f"{name}_token_loss"] = F.l1_loss(tk_predict, target_by_tokens)

            var = target.squeeze(-1)
        else:
            var = var_predict

        return (
            var,
            {
                f"{name}_vp_context": context,
                f"{name}_vp_context_mask": context_mask,
                f"{name}_vp_predict": var_predict.unsqueeze(-1),
                f"{name}_vp_target": kwargs.get("target"),
            },
            losses,
        )


class TokenLevelPredictorWithDiscriminatorParams(TokenLevelPredictorParams):
    pass


class TokenLevelPredictorWithDiscriminator(TokenLevelPredictor, Component):
    params: TokenLevelPredictorWithDiscriminatorParams

    def __init__(
        self,
        params: TokenLevelPredictorWithDiscriminatorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)
        self.disc = SignalDiscriminator(in_channels=params.vp_inner_dim)

    def forward_step(self, x, mask, **kwargs):
        var_predict, var_content, var_losses = super().forward_step(x, mask, **kwargs)

        var = kwargs.get("target")
        name = kwargs.get("name")
        m_inputs = kwargs.get("model_inputs")

        if self.training:
            var_real = var.unsqueeze(-1)
            var_fake = var_content[f"{name}_vp_predict"]
            context = var_content[f"{name}_vp_context"]
            context_mask = var_content[f"{name}_vp_context_mask"]

            disc_losses = self.disc.calculate_loss(
                context.transpose(1, -1),
                context_mask.transpose(1, -1),
                var_real.transpose(1, -1),
                var_fake.transpose(1, -1),
                m_inputs.global_step,
            )
            var_losses.update({f"{name}_{k}": v for k, v in disc_losses.items()})

        return var_predict, var_content, var_losses
