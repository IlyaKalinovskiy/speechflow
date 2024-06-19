import typing as tp

import torch

from pydantic import Field
from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import get_lengths_from_mask
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = [
    "FrameLevelPredictor",
    "FrameLevelPredictorParams",
    "FrameLevelPredictorWithDiscriminator",
    "FrameLevelPredictorWithDiscriminatorParams",
]


class FrameLevelPredictorParams(VariancePredictorParams):
    frame_encoder_type: str = "RNNEncoder"
    frame_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    use_ssl_adjustment: bool = False


class FrameLevelPredictor(Component):
    params: FrameLevelPredictorParams

    def __init__(
        self,
        params: FrameLevelPredictorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        def _init_encoder(_enc_cls, _enc_params_cls, _encoder_params, _input_dim):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = 1
            return _enc_cls(_enc_params, _input_dim)

        enc_cls, enc_params_cls = TTS_ENCODERS[params.frame_encoder_type]
        self.frame_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.frame_encoder_params, input_dim
        )

        if params.use_ssl_adjustment:
            self.ssl_adjustment = _init_encoder(
                enc_cls, enc_params_cls, params.frame_encoder_params, params.ssl_feat_dim
            )
            self.ssl_proj = Regression(params.vp_inner_dim * 2, 1)
        else:
            self.ssl_adjustment = None

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_mask, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")
        target = kwargs.get("target")

        m_inputs = kwargs.get("model_inputs")
        x_lengths = get_lengths_from_mask(x_mask)

        enc_predict, enc_ctx = self.frame_encoder.process_content(x, x_lengths, m_inputs)

        losses = {}
        content = {
            f"{name}_vp_context": enc_ctx,
            f"{name}_vp_context_mask": x_mask,
            f"{name}_vp_predict": enc_predict,
        }

        if self.training:
            if self.ssl_adjustment is not None:
                _, ssl_ctx = self.ssl_adjustment.process_content(
                    m_inputs.ssl_feat, x_lengths, m_inputs
                )
                var_from_ssl = self.ssl_proj(
                    torch.cat([enc_ctx.detach(), ssl_ctx], dim=2)
                )
                losses[f"{name}_ssl_adjustment_loss"] = F.l1_loss(
                    var_from_ssl, target.unsqueeze(-1)
                )
                var_by_frames = var_from_ssl.squeeze(-1).detach()
            else:
                var_by_frames = target

            if enc_predict is not None and var_by_frames is not None:
                if var_by_frames.ndim == 2:
                    var_by_frames = var_by_frames.unsqueeze(-1)

                losses[f"{name}_loss_by_frames"] = F.mse_loss(enc_predict, var_by_frames)
                content[f"{name}_vp_target"] = var_by_frames

        return enc_predict.squeeze(-1), content, losses


class FrameLevelPredictorWithDiscriminatorParams(FrameLevelPredictorParams):
    pass


class FrameLevelPredictorWithDiscriminator(FrameLevelPredictor, Component):
    params: FrameLevelPredictorWithDiscriminatorParams

    def __init__(
        self,
        params: FrameLevelPredictorWithDiscriminatorParams,
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
