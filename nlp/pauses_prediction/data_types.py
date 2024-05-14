from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.collate_functions.tts_collate import LinguisticFeatures
from speechflow.data_pipeline.core.datasample import Detachable, MovableToDevice

__all__ = [
    "PausesPredictionTarget",
    "PausesPredictionInput",
    "PausesPredictionOutput",
]


@dataclass
class PausesPredictionTarget(MovableToDevice, Detachable):
    durations: Tensor = None


@dataclass
class PausesPredictionInput(MovableToDevice, Detachable):
    transcription: Tensor = None
    ling_feat: LinguisticFeatures = None  # type: ignore
    durations: Tensor = None
    spectrogram: Tensor = None
    sil_masks: Tensor = None
    input_lengths: Tensor = None
    n_speakers: Tensor = None
    speaker_embedding: Tensor = None
    speaker_ids: Tensor = None


@dataclass
class PausesPredictionOutput(MovableToDevice, Detachable):
    durations: Tensor = None
    sil_masks: Tensor = None
