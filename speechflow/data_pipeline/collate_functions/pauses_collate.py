import typing as tp

from dataclasses import dataclass

from speechflow.data_pipeline.collate_functions.tts_collate import (
    TTSCollate,
    TTSCollateOutput,
)
from speechflow.data_pipeline.collate_functions.utils import collete_1d
from speechflow.data_pipeline.datasample_processors.data_types import (
    PausesPredictionDataSample,
)

__all__ = ["PausesPredictionCollate", "PausesPredictionCollateOutput"]


@dataclass
class PausesPredictionCollateOutput(TTSCollateOutput, PausesPredictionDataSample):
    pass


class PausesPredictionCollate(TTSCollate):
    def collate(  # type: ignore
        self, batch: tp.List[PausesPredictionDataSample]
    ) -> PausesPredictionCollateOutput:
        tts_collated = super().collate(batch)  # type: ignore
        collated = PausesPredictionCollateOutput(**tts_collated.to_dict())  # type: ignore

        collated.sil_mask, _ = collete_1d(
            batch, "sil_mask", self.pad_values, self.multiple_values
        )
        return collated
