import typing as tp

from dataclasses import dataclass

import torch
import numpy.typing as npt

from multilingual_text_parser.data_types import Sentence
from torch import Tensor

from speechflow.data_pipeline.core.datasample import (
    DataSample,
    MovableToDevice,
    ToNumpy,
    ToTensor,
)
from speechflow.io import AudioChunk, Timestamps

__all__ = [
    "ImageDataSample",
    "AudioDataSample",
    "SpectrogramDataSample",
    "TextDataSample",
    "TTSDataSample",
    "PausesPredictionDataSample",
    "SSLFeatures",
    "AudioCodecFeatures",
    "ProsodySSMLDataSample",
    "ProsodyPredictionDataSample",
]

tp_DATA = tp.Union[npt.NDArray, Tensor]


@dataclass(eq=False)
class ImageDataSample(DataSample):
    image: tp_DATA = None


@dataclass
class SSLFeatures(ToTensor, ToNumpy, MovableToDevice):
    encode: tp_DATA = None  # type: ignore
    projection: tp_DATA = None  # type: ignore
    attention_mask: tp_DATA = None  # type: ignore

    def __getitem__(self, item):
        return self.encode[item]

    def get(self):
        return self.encode


@dataclass
class AudioCodecFeatures(ToTensor, ToNumpy, MovableToDevice):
    encode: tp_DATA = None  # type: ignore
    waveform: tp_DATA = None  # type: ignore

    def __getitem__(self, item):
        return self.encode[item]

    def get(self):
        return self.encode


@dataclass(eq=False)
class AudioDataSample(DataSample):
    audio_chunk: AudioChunk = None  # type: ignore
    lang: str = None  # type: ignore
    lang_id: int = None  # type: ignore
    speaker_name: str = None  # type: ignore
    speaker_id: int = 0
    speaker_emb: tp_DATA = None  # type: ignore
    speaker_emb_mean: tp_DATA = None  # type: ignore
    speech_quality_emb: tp_DATA = None  # type: ignore
    lpc_feat: tp_DATA = None  # type: ignore
    ssl_feat: SSLFeatures = None  # type: ignore
    ac_feat: AudioCodecFeatures = None  # type: ignore
    mu_law_waveform: tp_DATA = None  # type: ignore
    lpc_waveform: tp_DATA = None  # type: ignore
    bits: int = 16

    def __len__(self):
        if self.audio_chunk and self.audio_chunk.duration:
            return int(self.audio_chunk.duration * 1000)  # in milliseconds
        else:
            return 0

    def __lt__(self, other):
        return len(self) < len(other)


@dataclass(eq=False)
class SpectrogramDataSample(AudioDataSample):
    magnitude: tp_DATA = None  # type: ignore
    mel: tp_DATA = None  # type: ignore
    energy: tp_DATA = None  # type: ignore
    spectral_flatness: tp_DATA = None  # type: ignore
    spectral_tilt: tp_DATA = None  # type: ignore
    spectral_envelope: tp_DATA = None  # type: ignore
    pitch: tp_DATA = None  # type: ignore
    precomputed_mel: tp_DATA = None  # type: ignore
    averages: tp.Dict[str, tp_DATA] = None  # type: ignore
    ranges: tp.Dict[str, tp_DATA] = None  # type: ignore
    gate: tp_DATA = None  # type: ignore

    def __len__(self):
        if self.magnitude is not None:
            return self.magnitude.shape[0]
        elif self.audio_chunk:
            return super().__len__()
        else:
            return 0


@dataclass(eq=False)
class TextDataSample(DataSample):
    sent: Sentence = None  # type: ignore
    symbols: tp.Tuple[str, ...] = None  # type: ignore
    transcription: tp_DATA = None  # type: ignore
    ling_feat: tp.Dict[str, tp_DATA] = None  # type: ignore
    intonation_type: int = None  # type: ignore
    word_lengths: tp_DATA = None  # type: ignore
    synt_lengths: tp_DATA = None  # type: ignore
    lm_feat: tp_DATA = None  # type: ignore
    pad_symb_id: int = 0
    sil_symb_id: int = 0


@dataclass(eq=False)
class ProsodySSMLDataSample(DataSample):
    temp_modifier: tp_DATA = None  # type: ignore
    pitch_modifier: tp_DATA = None  # type: ignore
    volume_modifier: tp_DATA = None  # type: ignore


@dataclass(eq=False)
class TTSDataSample(SpectrogramDataSample, TextDataSample, ProsodySSMLDataSample):
    word_timestamps: Timestamps = None  # type: ignore
    phoneme_timestamps: tp.List[Timestamps] = None  # type: ignore
    durations: tp_DATA = None  # type: ignore
    invert_durations: tp_DATA = None  # type: ignore
    transcription_by_frames: tp_DATA = None
    aggregated: tp.Dict[str, tp_DATA] = None  # type: ignore
    pauses_durations: torch.Tensor = None  # type: ignore

    def __str__(self) -> str:
        return self.sent.text_orig if self.sent else ""


@dataclass(eq=False)
class PausesPredictionDataSample(TTSDataSample):
    sil_mask: tp_DATA = None  # type: ignore


@dataclass(eq=False)
class ProsodyPredictionDataSample(TTSDataSample):
    attention_mask: tp_DATA = None  # type: ignore
    input_ids: tp_DATA = None  # type: ignore
    binary: tp_DATA = None  # type: ignore
    category: tp_DATA = None  # type: ignore
    pad_id: int = None  # type: ignore
    lang: str = None  # type: ignore
    word_ids: tp_DATA = None  # type: ignore
    seed_by_words: tp.List[int] = None  # type: ignore

    def __len__(self) -> int:
        return self.input_ids.shape[0]
