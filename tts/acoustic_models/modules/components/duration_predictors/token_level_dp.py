import typing as tp

import torch

from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.components.variance_predictors.tokenlevel_predictor import (
    TokenLevelPredictor,
    TokenLevelPredictorParams,
)

__all__ = [
    "TokenLevelDP",
    "TokenLevelDPParams",
]


class TokenLevelDPParams(TokenLevelPredictorParams):
    activation_fn: str = "SiLU"
    loss_type: str = "cross_entropy"
    discrete_scale: float = 1.0
    add_noise: bool = False
    noise_scale: float = 0.1
    every_iter: int = 2
    beta: float = 20.0


class TokenLevelDP(TokenLevelPredictor):
    params: TokenLevelDPParams

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.params.loss_type == "cross_entropy":
            assert self.params.variance_params.dim == 1  # support only 1D variances
            assert self.params.vp_output_dim > 1

    @property
    def output_dim(self):
        return 1  # support only 1D variances

    def postprocessing(self, predict):
        if self.params.loss_type == "cross_entropy":
            predict[:, :, :-1] = torch.sigmoid(predict[:, :, :-1]) > 0.5
            predict = predict.sum(dim=-1) / self.params.discrete_scale

        if self.params.variance_params.log_scale:  # type: ignore
            predict = torch.expm1(predict)

        return predict

    def compute_loss(
        self, name, predict, target, lengths, global_step: int = 0
    ) -> tp.Dict[str, torch.Tensor]:
        loss_fn = getattr(F, self.params.loss_type)
        losses = {}

        if global_step % self.params.every_iter != 0:
            return losses

        if self.params.variance_params.log_scale:  # type: ignore
            target = torch.log1p(target)

        if self.params.add_noise:
            target = (target + self.params.noise_scale * torch.randn_like(target)).clip(
                min=0
            )

        target = apply_mask(target, get_mask_from_lengths(lengths))
        predict = apply_mask(predict, get_mask_from_lengths(lengths))

        if self.params.loss_type == "cross_entropy":
            target *= self.params.discrete_scale

            reg_loss = 0
            ce_loss = 0
            for dur, enc, dlen in zip(target, predict, lengths):
                dur = dur[:dlen]
                enc = enc[:dlen, :]
                trg = torch.zeros_like(enc)

                trunc = torch.trunc(dur)
                frac = torch.frac(dur)

                cols = torch.arange(trg.shape[1], device=trg.device)
                mask = cols < trunc.long().unsqueeze(1)

                trg[mask] = 1
                trg[:, -1] = frac

                dur_pred = torch.sigmoid(enc[:, :-1]).sum(dim=-1)
                reg_loss += F.l1_loss(dur_pred, trunc) + F.l1_loss(enc[:, -1], frac)

                ce_loss += F.binary_cross_entropy_with_logits(enc[:, :-1], trg[:, :-1])

            losses[f"{name}_cross_entropy_loss"] = (
                reg_loss + self.params.beta * ce_loss
            ) / predict.shape[0]
        else:
            losses[f"{name}_{self.params.loss_type}"] = loss_fn(predict, target)

        return losses
