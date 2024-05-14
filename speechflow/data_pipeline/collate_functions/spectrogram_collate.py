import typing as tp

from dataclasses import dataclass

import numpy as np
import torch

from torch import Tensor

from speechflow.data_pipeline.collate_functions.audio_collate import (
    AudioCollate,
    AudioCollateOutput,
)
from speechflow.data_pipeline.datasample_processors.data_types import (
    SpectrogramDataSample,
)
from speechflow.training.utils.pad_utils import pad, pad_2d, sequence_collate

__all__ = [
    "SpectrogramCollate",
    "SpectrogramCollateOutput",
]


@dataclass
class SpectrogramCollateOutput(AudioCollateOutput):
    linear_spectrogram: Tensor = None
    mel_spectrogram: Tensor = None
    gate: Tensor = None
    energy: Tensor = None
    spectral_flatness: Tensor = None
    spectral_envelope: Tensor = None
    pitch: Tensor = None
    averages: tp.Dict[str, Tensor] = None  # type: ignore
    ranges: tp.Dict[str, Tensor] = None  # type: ignore
    spectrogram_lengths: Tensor = None

    def __post_init__(self):
        super().__post_init__()

        if self.averages is None:
            self.averages = {}


class SpectrogramCollate(AudioCollate):
    def __call__(self, batch: tp.List[SpectrogramDataSample]) -> SpectrogramCollateOutput:  # type: ignore
        audio_collated = super().__call__(batch)  # type: ignore
        collated = SpectrogramCollateOutput(**audio_collated.to_dict())  # type: ignore
        pad_val = self.pad_values.get("spec", 0)
        multiple = self.multiple.get("spec")

        if batch[0].magnitude is not None:
            linear_spectrogram = [sample.magnitude for sample in batch]  # type: ignore
            height = linear_spectrogram[0].size(1)
            min_level_db = batch[0].get_param_val("min_level_db", pad_val)
            linear_spectrogram, spec_lens = pad_2d(
                linear_spectrogram, min_level_db, height, multiple=multiple
            )
        else:
            linear_spectrogram = None
            spec_lens = None

        if batch[0].mel is not None:
            mel_spectrogram = [sample.mel for sample in batch]  # type: ignore
            height = mel_spectrogram[0].size(1)
            mel_min_val = batch[0].get_param_val("mel_min_val", pad_val)
            mel_spectrogram, spec_lens = pad_2d(
                mel_spectrogram, mel_min_val, height, multiple=multiple
            )
        else:
            mel_spectrogram = None

        gate, _ = sequence_collate(batch, "gate", multiple=multiple)

        if batch[0].energy is not None:
            energy = [sample.energy for sample in batch]
            energy, en_lens = pad(
                energy, self.pad_values.get("energy", 0.0), multiple=multiple
            )
        else:
            energy = None
            en_lens = None

        if batch[0].spectral_flatness is not None:
            spectral_flatness = [sample.spectral_flatness for sample in batch]
            spectral_flatness, sf_lens = pad(
                spectral_flatness,
                self.pad_values.get("spectral_flatness", 0.0),
                multiple=multiple,
            )
        else:
            spectral_flatness = None
            sf_lens = None

        if batch[0].spectral_envelope is not None:
            spectral_envelope = [sample.spectral_envelope for sample in batch]  # type: ignore
            height = spectral_envelope[0].size(1)
            spectral_envelope, env_lens = pad_2d(
                spectral_envelope, height=height, multiple=multiple
            )
        else:
            spectral_envelope = None
            env_lens = None

        if batch[0].pitch is not None:
            if batch[0].pitch.ndim == 1:
                pitch = [sample.pitch for sample in batch]
                pitch, pitch_lens = pad(
                    pitch, self.pad_values.get("pitch", 0.0), multiple=multiple
                )
            else:
                pitch = [sample.pitch for sample in batch]  # type: ignore
                pitch, pitch_lens = pad_2d(
                    pitch, height=pitch[0].shape[1], multiple=multiple
                )
        else:
            pitch = None
            pitch_lens = None

        if batch[0].averages is not None:
            averages = {}
            for name in batch[0].averages.keys():
                averages[name] = torch.tensor(
                    [[sample.averages[name]] for sample in batch]
                )
        else:
            averages = None

        if batch[0].ranges is not None:
            ranges = {}
            for name in batch[0].ranges.keys():
                ranges[name] = torch.from_numpy(
                    np.stack([sample.ranges[name] for sample in batch])
                )
        else:
            ranges = None

        for lens in [
            spec_lens,
            en_lens,
            sf_lens,
            env_lens,
            pitch_lens,
        ]:
            if spec_lens is not None and lens is not None:
                assert (spec_lens == torch.LongTensor(lens)).all()

        collated.linear_spectrogram = linear_spectrogram
        collated.mel_spectrogram = mel_spectrogram
        collated.gate = gate
        collated.energy = energy
        collated.spectral_flatness = spectral_flatness
        collated.spectral_envelope = spectral_envelope
        collated.pitch = pitch
        collated.averages = averages
        collated.ranges = ranges
        collated.spectrogram_lengths = spec_lens
        return collated
