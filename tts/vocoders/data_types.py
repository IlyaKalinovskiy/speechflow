import typing as tp

from dataclasses import dataclass

import torch

from speechflow.data_pipeline.core import TrainData
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput

__all__ = [
    "VocoderTarget",
    "VocoderForwardInput",
    "VocoderForwardOutput",
    "VocoderInferenceInput",
    "VocoderInferenceOutput",
]


@dataclass
class VocoderTarget(TrainData):
    phase: str = None  # type: ignore


@dataclass
class VocoderForwardInput(TTSForwardInput):
    lpc: torch.Tensor = None
    lpc_feat: torch.Tensor = None


@dataclass
class VocoderForwardOutput(TrainData):
    waveform: torch.Tensor = None
    waveform_length: torch.Tensor = None
    additional_content: tp.Dict[str, torch.Tensor] = None

    def __post_init__(self):
        if self.additional_content is None:
            self.additional_content = {}


@dataclass
class VocoderInferenceInput(VocoderForwardInput):
    @staticmethod
    def init_from_tts_output(tts_output: TTSForwardOutput) -> "VocoderInferenceInput":
        pass


@dataclass
class VocoderInferenceOutput(VocoderForwardOutput):
    pass
