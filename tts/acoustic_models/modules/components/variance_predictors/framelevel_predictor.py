import typing as tp

import torch

from pydantic import Field
from torch.nn import functional as F

from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.common import VarianceEmbedding
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.components.discriminators import SignalDiscriminator
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = [
    "FrameLevelPredictor",
    "FrameLevelPredictorParams",
    "FrameLevelPredictorWithDiscriminator",
    "FrameLevelPredictorWithDiscriminatorParams",
]


class FrameLevelPredictorParams(VariancePredictorParams):
    frame_encoder_type: str = "VarianceEncoder"
    frame_encoder_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})
    use_ssl_adjustment: bool = (
        False  # improving the target feature through prediction over SSL model
    )
    use_mtm: bool = False  # masked token modeling
    var_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class FrameLevelPredictor(Component):
    params: FrameLevelPredictorParams

    def __init__(
        self,
        params: FrameLevelPredictorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

        from tts.acoustic_models.modules import TTS_ENCODERS

        def _init_encoder(
            _enc_cls, _enc_params_cls, _encoder_params, _input_dim, _output_dim=1
        ):
            _enc_params = _enc_params_cls.init_from_parent_params(params, _encoder_params)
            _enc_params.encoder_num_blocks = params.vp_num_blocks
            _enc_params.encoder_num_layers = params.vp_num_layers
            _enc_params.encoder_inner_dim = params.vp_inner_dim
            _enc_params.encoder_output_dim = _output_dim
            return _enc_cls(_enc_params, _input_dim)

        enc_cls, enc_params_cls = TTS_ENCODERS[params.frame_encoder_type]

        if params.use_ssl_adjustment:
            self.ssl_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.frame_encoder_params, params.ssl_feat_dim
            )
            self.ssl_proj = Regression(params.vp_inner_dim * 2, 1)
        else:
            self.ssl_encoder = None

        if params.use_mtm:
            emb_dim = params.var_params.emb_dim  # type: ignore
            self.mtm_embeddings = VarianceEmbedding(
                interval=(0, params.var_params.interval[1]),  # type: ignore
                n_bins=params.var_params.n_bins,  # type: ignore
                log_scale=params.var_params.log_scale,  # type: ignore
                emb_dim=emb_dim,  # type: ignore
            )
            self.mtm_encoder = _init_encoder(
                enc_cls, enc_params_cls, params.frame_encoder_params, 2 * emb_dim, emb_dim
            )
            self.mtm_proj = Regression(input_dim, emb_dim)

            input_dim += 2 * emb_dim
        else:
            self.mtm_encoder = None

        self.frame_encoder = _init_encoder(
            enc_cls, enc_params_cls, params.frame_encoder_params, input_dim
        )

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        name = kwargs.get("name")
        target = kwargs.get("target")
        losses = {}

        if self.mtm_encoder is not None and target is not None:
            target_mask = target

            mtm_embs = self.mtm_embeddings(target_mask)
            mtm_x = torch.cat([mtm_embs, self.mtm_proj(x.detach())], dim=-1)
            mtm_predict, _ = self.mtm_encoder.process_content(
                mtm_x, x_lengths, model_inputs
            )

            if self.training:
                losses[f"{name}_mtm_loss"] = F.mse_loss(
                    mtm_predict, self.mtm_embeddings(target).detach()
                )

            if model_inputs.imputer_masks is not None:
                mtm_predict = mtm_predict * (
                    ~model_inputs.imputer_masks["spectrogram"]
                ).unsqueeze(-1)

            x = torch.cat([x, mtm_embs, mtm_predict], dim=-1)

        enc_predict, enc_ctx = self.frame_encoder.process_content(
            x, x_lengths, model_inputs
        )

        content = {
            f"{name}_vp_context": enc_ctx,
            f"{name}_vp_predict": enc_predict,
        }

        if self.training:
            if self.ssl_encoder is not None:
                _, ssl_ctx = self.ssl_encoder.process_content(
                    model_inputs.ssl_feat, x_lengths, model_inputs
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

                losses[f"{name}_loss_by_frames"] = F.l1_loss(enc_predict, var_by_frames)
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

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        var_predict, var_content, var_losses = super().forward_step(
            x, x_lengths, model_inputs, **kwargs
        )

        var = kwargs.get("target")
        name = kwargs.get("name")

        if self.training:
            var_real = var.unsqueeze(-1)
            var_fake = var_content[f"{name}_vp_predict"]
            context = var_content[f"{name}_vp_context"]
            mask = get_mask_from_lengths(x_lengths)

            disc_losses = self.disc.calculate_loss(
                context.transpose(1, -1),
                mask.transpose(1, -1),
                var_real.transpose(1, -1),
                var_fake.transpose(1, -1),
                model_inputs.global_step,
            )
            var_losses.update({f"{name}_{k}": v for k, v in disc_losses.items()})

        return var_predict, var_content, var_losses
