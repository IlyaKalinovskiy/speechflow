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
    activation_fn: str = "ReLU"
    loss_type: str = "l1_loss"
    add_noise: bool = False
    deterministic: bool = False
    every_iter: int = 2


class TokenLevelDP(TokenLevelPredictor):
    params: TokenLevelDPParams

    def postprocessing(self, predict):
        if self.params.deterministic:
            predict = (torch.sigmoid(predict) > 0.5).sum(dim=-1).float()
        else:
            if self.params.var_params.log_scale:  # type: ignore
                predict = torch.expm1(predict)

        return predict

    def compute_loss(
        self, name, predict, target, lengths, global_step: int = 0
    ) -> tp.Dict[str, torch.Tensor]:
        loss_fn = getattr(F, self.params.loss_type)
        losses = {}

        if global_step % self.params.every_iter != 0:
            return losses

        if self.params.add_noise:
            target = (target + 0.1 * torch.randn_like(target)).clip(min=0.1)

        if self.params.deterministic:
            reg_loss = 0
            ce_loss = 0
            for dur, enc, dlen in zip(target, predict, lengths):
                enc = enc[:dlen, :]
                dur = dur[:dlen]
                trg = torch.zeros_like(enc)
                for p in range(trg.shape[0]):
                    trunc = torch.trunc(dur[p]).long()
                    frac = torch.frac(dur[p])
                    trg[p, :trunc] = 1
                    trg[p, -1] = frac

                dur_pred = torch.sigmoid(enc[:, :-1]).sum(dim=-1)
                if self.params.var_params.log_scale:
                    reg = loss_fn(torch.log(dur_pred), torch.log(dur))
                else:
                    reg = loss_fn(dur_pred, dur)

                reg_loss += reg
                reg_loss += loss_fn(enc[:, -1], trg[:, -1])

                ce = F.binary_cross_entropy_with_logits(
                    enc[:, :-1].flatten(), trg[:, :-1].flatten()
                )
                ce_loss += ce

            losses[f"{name}_deterministic_loss"] = (
                reg_loss + 20 * ce_loss
            ) / predict.shape[0]
        else:
            if self.params.var_params.log_scale:  # type: ignore
                mask = get_mask_from_lengths(lengths)
                target = apply_mask(torch.log(target), mask)

            losses[f"{name}_{self.params.loss_type}"] = loss_fn(predict, target)

        return losses
