import typing as tp

from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.collate_functions.tts_collate import (
    TTSCollate,
    TTSCollateOutput,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    PausesPredictionDataSample,
)
from speechflow.training.utils.pad_utils import pad

__all__ = ["PausesPredictionCollate", "PausesPredictionCollateOutput"]


@dataclass
class PausesPredictionCollateOutput(TTSCollateOutput):
    sil_masks: tp.Optional[Tensor] = None


class PausesPredictionCollate(TTSCollate):
    def __call__(  # type: ignore
        self, batch: tp.List[PausesPredictionDataSample]
    ) -> PausesPredictionCollateOutput:
        tts_collated = super().__call__(batch)  # type: ignore
        collated = PausesPredictionCollateOutput(**tts_collated.to_dict())  # type: ignore

        if batch[0].sil_mask is not None:
            sil_masks = [sample.sil_mask for sample in batch]
            sil_masks, _ = pad(sil_masks)
        else:
            sil_masks = None

        collated.sil_masks = sil_masks
        return collated
