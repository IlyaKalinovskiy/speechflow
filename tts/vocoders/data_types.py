import typing as tp

from dataclasses import dataclass

import torch

from speechflow.data_pipeline.core import TrainData

__all__ = [
    "VocoderTarget",
    "VocoderForwardInput",
    "VocoderForwardOutput",
    "VocoderInferenceOutput",
]


@dataclass
class VocoderTarget(TrainData):
    phase: str = None  # type: ignore


@dataclass
class VocoderForwardInput(TrainData):
    lang_id: torch.Tensor = None
    speaker_embedding: torch.Tensor = None
    spectrogram: torch.Tensor = None
    spectrogram_lengths: torch.Tensor = None
    linear_spectrogram: torch.Tensor = None
    linear_spectrogram_lengths: torch.Tensor = None
    lpc: torch.Tensor = None
    lpc_feat: torch.Tensor = None
    ssl_embeddings: torch.Tensor = None
    ssl_embeddings_lengths: torch.Tensor = None
    energy: torch.Tensor = None
    pitch: torch.Tensor = None
    additional_inputs: tp.Dict = None


@dataclass
class VocoderForwardOutput(TrainData):
    pass


@dataclass
class VocoderInferenceOutput(TrainData):
    # Waveform in shape (B, T)
    waveform: torch.Tensor = None
    spectrogram: torch.Tensor = None
    additional_content: tp.Dict[str, torch.Tensor] = None
    # TODO: Information about audio lengths and pad lengths

    def __post_init__(self):
        if self.additional_content is None:
            self.additional_content = {}
