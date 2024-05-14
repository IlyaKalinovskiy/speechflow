from torch import nn

from nlp.pauses_prediction.data_types import (
    PausesPredictionOutput,
    PausesPredictionTarget,
)

__all__ = ["PausesPredictionLoss"]


class PausesPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss()

    def forward(
        self,
        current_iter: int,
        output: PausesPredictionOutput,
        target: PausesPredictionTarget,
    ) -> dict:

        predicted_durations = output.durations.squeeze(-1)
        target_durations = target.durations.float()
        sil_masks = output.sil_masks

        _is_sil = sil_masks > 0
        predicted_sil = predicted_durations.masked_select(_is_sil)
        target_sil = target_durations.masked_select(_is_sil)

        sil_loss = self.mse_loss_fn(predicted_sil, target_sil)

        return {"SIL MSE": sil_loss}
