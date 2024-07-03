import typing as tp

import torch

from pydantic import Field
from torch import nn
from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.common import SoftLengthRegulator
from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = [
    "TokenLevelPredictor",
    "TokenLevelPredictorParams",
    "TokenLevelPredictorWithDiscriminator",
    "TokenLevelPredictorWithDiscriminatorParams",
]


class TokenLevelPredictorParams(VariancePredictorParams):
    word_encoder_type: str = "VarianceEncoder"
    word_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    token_encoder_type: str = "VarianceEncoder"
    token_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    add_lm_feat: bool = False


class TokenLevelPredictor(Component):
    params: TokenLevelPredictorParams

    def __init__(
        self,
        params: TokenLevelPredictorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        def _init_encoder(_enc_cls, _enc_params_cls, _encoder_params):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = params.vp_output_dim
            return _enc_cls(_enc_params, input_dim)

        enc_cls, enc_params_cls = TTS_ENCODERS[params.token_encoder_type]
        self.token_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.token_encoder_params
        )
        self.lr = SoftLengthRegulator()

        if params.add_lm_feat:
            self.lm_proj = nn.Linear(params.lm_feat_dim, input_dim, bias=False)
            self.token_proj = nn.Linear(
                input_dim + params.vp_inner_dim, input_dim, bias=False
            )

            enc_cls, enc_params_cls = TTS_ENCODERS[params.word_encoder_type]
            self.word_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.word_encoder_params
            )

            self.hard_lr = SoftLengthRegulator(hard=True)
        else:
            self.token_proj = nn.Identity()

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")

        losses = {}

        word_length = model_inputs.additional_inputs.get("word_lengths")
        word_inv_lengths = model_inputs.additional_inputs.get("word_invert_lengths")

        max_num_tokens = model_inputs.transcription_lengths.max().item()
        max_num_words = model_inputs.num_words.max().item()

        if self.params.add_lm_feat:
            x_by_words, _ = self.lr(x.detach(), word_inv_lengths, max_num_words)
            x_by_words = x_by_words + self.lm_proj(model_inputs.lm_feat)
            wd_predict, wd_ctx = self.word_encoder.process_content(
                x_by_words, model_inputs.num_words, model_inputs
            )
            wd_ctx, _ = self.hard_lr(wd_ctx, word_length, max_num_tokens)
            x_by_tokens = torch.cat([x, wd_ctx], dim=2)
        else:
            x_by_tokens = x

        tk_proj = self.token_proj(x_by_tokens)
        tk_predict, tk_ctx = self.token_encoder.process_content(
            tk_proj, model_inputs.input_lengths, model_inputs
        )

        context = tk_ctx
        var_predict = tk_predict.squeeze(-1)

        if self.training:
            target_by_tokens = kwargs.get("target")
            if target_by_tokens.ndim == 2:
                target_by_tokens = target_by_tokens.unsqueeze(-1)

            losses[f"{name}_loss_by_tokens"] = F.l1_loss(tk_predict, target_by_tokens)

        return (
            var_predict.squeeze(-1),
            {
                f"{name}_vp_context": context,
                f"{name}_vp_predict": var_predict,
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
        if params.vp_output_dim != 1:
            raise ValueError("Feature dimension must be equal to 1")

        super().__init__(params, input_dim)
        self.disc = SignalDiscriminator(in_channels=params.vp_inner_dim)

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        var_predict, var_content, var_losses = super().forward_step(
            x, x_lengths, model_inputs, **kwargs
        )

        var = kwargs.get("target")
        name = kwargs.get("name")

        if self.training:
            var_real = var
            var_fake = var_content[f"{name}_vp_predict"]
            context = var_content[f"{name}_vp_context"]
            mask = get_mask_from_lengths(x_lengths).unsqueeze(-1)

            if var_real.ndim == 2:
                var_real = var_real.unsqueeze(-1)
            if var_fake.ndim == 2:
                var_fake = var_fake.unsqueeze(-1)

            disc_losses = self.disc.calculate_loss(
                context.transpose(1, -1),
                mask.transpose(1, -1),
                var_real.transpose(1, -1),
                var_fake.transpose(1, -1),
                model_inputs.global_step,
            )
            var_losses.update({f"{name}_{k}": v for k, v in disc_losses.items()})

        return var_predict, var_content, var_losses
