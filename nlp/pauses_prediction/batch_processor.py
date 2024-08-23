import typing as tp

from nlp.pauses_prediction.data_types import PausesPredictionInput, PausesPredictionTarget
from speechflow.data_pipeline.collate_functions import PausesPredictionCollateOutput
from speechflow.data_pipeline.core import BaseBatchProcessor, Batch, DataSample

__all__ = ["PausesPredictionProcessor"]


class PausesPredictionProcessor(BaseBatchProcessor):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        batch: Batch,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Tuple[
        PausesPredictionInput, PausesPredictionTarget, tp.Optional[tp.List[DataSample]]
    ]:
        collated: PausesPredictionCollateOutput = batch.collated_samples  # type: ignore
        _input: PausesPredictionInput = PausesPredictionInput(
            transcription=collated.transcription_id,
            ling_feat=collated.ling_feat,
            durations=collated.durations,
            sil_mask=collated.sil_mask,
            input_lengths=collated.transcription_lengths,
            speaker_ids=collated.speaker_ids,
        )
        _target: PausesPredictionTarget = PausesPredictionTarget(
            durations=collated.durations
        )

        return _input.to(self.device), _target.to(self.device), batch.data_samples
