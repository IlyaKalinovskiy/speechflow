import io
import typing as tp

from copy import deepcopy as copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pydub
import librosa
import soundfile as sf
import numpy.typing as npt

from scipy import signal

LIBROSA_VERSION = list(map(int, librosa.version.short_version.split(".")))

__all__ = ["AudioFormat", "AudioChunk"]


class AudioFormat(Enum):
    wav = 0
    mp3 = 1
    ogg = 2
    opus = 3

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @staticmethod
    def get_formats() -> tp.Tuple[str, ...]:
        return tuple([f".{name}" for name in AudioFormat.names()])


@dataclass
class AudioChunk:
    file_path: tp.Union[str, Path] = None  # type: ignore
    data: npt.NDArray = None  # type: ignore
    sr: int = None  # type: ignore
    begin: float = 0.0
    end: float = None  # type: ignore
    fade_duration: tp.Optional[tp.Tuple[float, float]] = None
    is_trim: bool = False

    def __post_init__(self):
        if self.file_path is not None:
            self.file_path = Path(self.file_path)
            assert self.file_path.exists() or self.data is not None, "wav file not found!"
        else:
            assert self.waveform is not None, "wave data not set!"

        self._set_end()

    def _set_end(self):
        if self.end is None:
            if self.waveform is None:
                assert self.file_path, "wav_path not set!"
                try:
                    self.sr = int(librosa.get_samplerate(path=self.file_path.as_posix()))
                    if LIBROSA_VERSION[1] <= 9:
                        self.end = librosa.get_duration(
                            filename=self.file_path.as_posix()
                        )
                    else:
                        self.end = librosa.get_duration(path=self.file_path.as_posix())
                except Exception:
                    self.load()
            else:
                self.end = len(self.waveform) / self.sr

    @property
    def waveform(self) -> npt.NDArray:
        return self.data

    @waveform.setter
    def waveform(self, waveform: npt.NDArray):
        assert len(waveform) == len(self.waveform)
        self.data = waveform

    @property
    def dtype(self):
        return self.waveform.dtype

    @property
    def empty(self):
        return self.waveform is None

    @property
    def duration(self) -> float:
        if self.end:
            return self.end - self.begin
        else:
            return 0.0

    @property
    def mean_volume(self) -> float:
        s = librosa.magphase(librosa.stft(self.data, window=np.ones, center=False))[0]
        return float(np.mean(librosa.feature.rms(S=s).T, axis=0))

    def load(
        self,
        sr: tp.Optional[int] = None,
        dtype: npt.DTypeLike = np.float32,
        load_entire_file: bool = False,
    ) -> "AudioChunk":
        assert isinstance(self.file_path, Path), "wav path not set!"
        assert self.file_path.exists(), f"wav file {self.file_path.as_posix()} not found!"

        if load_entire_file:
            self.data, self.sr = librosa.load(self.file_path, sr=sr)
            self.is_trim = False
        else:
            self.data, self.sr = librosa.load(
                self.file_path, sr=sr, offset=self.begin, duration=self.duration
            )
            if LIBROSA_VERSION[1] <= 9:
                self.is_trim = (
                    librosa.get_duration(filename=self.file_path.as_posix())
                    != self.duration
                )
            else:
                self.is_trim = (
                    librosa.get_duration(path=self.file_path.as_posix()) != self.duration
                )

        self._set_end()

        if self.fade_duration is not None:
            self._apply_fade(
                self.data, self.sr, self.fade_duration[0], self.fade_duration[1]
            )

        return self.astype(dtype, inplace=True)

    def save(
        self,
        audio_path: tp.Optional[tp.Union[str, Path, io.BytesIO]] = None,
        overwrite: bool = False,
    ):
        audio_path = audio_path if audio_path else self.file_path
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        if isinstance(audio_path, Path):
            audio_format = audio_path.suffix[1:]
            if audio_format not in ["wav", "flac"]:
                raise ValueError(f"'{audio_format}' audio format is not supported.")

            if not overwrite:
                if isinstance(audio_path, (Path, str)):
                    assert not Path(
                        audio_path
                    ).exists(), f"file {str(audio_path)} is exists!"
        else:
            audio_format = "wav"

        data = self.astype(np.float32).data

        if data.ndim == 2 and data.shape[0] == 1:
            raise ValueError(
                "Unacceptable data shape, single-channel data must be flatten."
            )

        sf.write(audio_path, data, self.sr, format=audio_format)
        return self

    def to_bytes(
        self, audio_format: AudioFormat = AudioFormat.wav, bitrate: str = "128k"
    ):
        buff = io.BytesIO()
        if audio_format in [AudioFormat.wav]:
            self.save(buff)
        elif audio_format in [AudioFormat.mp3, AudioFormat.ogg, AudioFormat.opus]:
            in_buff = io.BytesIO(self.astype(np.int16).data.tobytes())
            audio = pydub.AudioSegment.from_raw(
                in_buff, sample_width=2, channels=1, frame_rate=self.sr
            )
            buff = audio.export(
                buff,
                format=audio_format.name,
                codec="opus"
                if audio_format in [AudioFormat.ogg, AudioFormat.opus]
                else None,
                bitrate=None if audio_format == AudioFormat.ogg else bitrate,
                parameters=["-strict", "-2"],
            )
        else:
            raise NotImplementedError(f"Audio format {audio_format} is not supported")

        return buff.getvalue()

    def erase(self):
        del self.data
        return self

    def copy(self):
        return copy(self)

    def gsm_preemphasis(self, beta: float = 0.86, inplace: bool = False) -> "AudioChunk":
        """High-pass filter for telephone channel https://edadocs.software.keys
        ight.com/display/ads2009/GSM+Preemphasis."""
        sig = self.data
        assert np.issubdtype(
            self.data.dtype, np.floating
        ), "Audio data must be floating-point!"
        sig = signal.lfilter([1, -beta], [1], sig)
        if inplace:
            self.data = sig
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=self.sr,
                data=sig,
            )

    def trim(
        self,
        begin: tp.Optional[float] = None,
        end: tp.Optional[float] = None,
        inplace: bool = False,
    ) -> "AudioChunk":
        if begin is None and end is None:
            if self.is_trim:
                return AudioChunk(
                    begin=0.0, end=self.duration, sr=self.sr, data=self.data.copy()
                )
            else:
                return self if inplace else copy(self)

        begin = int(begin * self.sr) if begin else 0
        end = int(end * self.sr) if end else len(self.data)
        end = min(end, len(self.data))
        assert begin >= 0 and end <= len(self.data)
        assert begin < end

        if inplace:
            assert not self.is_trim, "wave is already trimmed!"
            self.begin = 0
            self.end = (end - begin) / self.sr
            self.data = self.data[begin:end]
            self.is_trim = True
            return self
        else:
            return AudioChunk(
                begin=0.0,
                end=(end - begin) / self.sr,
                sr=self.sr,
                data=self.data[begin:end],
            )

    def volume(self, value: float = 1.0, inplace: bool = False):
        sig = self.data
        assert np.issubdtype(
            self.data.dtype, np.floating
        ), "Audio data must be floating-point!"
        if value != 1.0:
            sig = np.clip(sig * value, a_min=-1.0, a_max=1.0)
        if inplace:
            self.data = sig
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=self.sr,
                data=sig,
            )

    def resample(
        self, sr: int, inplace: bool = False, fast: bool = False
    ) -> "AudioChunk":
        if self.sr != sr:
            if fast:
                data = librosa.resample(
                    self.data, orig_sr=self.sr, target_sr=sr, res_type="kaiser_fast"
                )
            else:
                data = librosa.resample(self.data, orig_sr=self.sr, target_sr=sr)
        else:
            data = self.data if inplace else self.data.copy()

        if inplace:
            self.sr = sr
            self.data = data
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=sr,
                data=data,
            )

    def astype(self, dtype, inplace: bool = False) -> "AudioChunk":
        data = self.data

        if self.dtype != dtype:
            if all(
                np.issubdtype(dt, np.signedinteger) for dt in [self.dtype, dtype]
            ) or all(np.issubdtype(dt, np.floating) for dt in [self.dtype, dtype]):
                data = self.data.astype(dtype)
            else:
                scale = np.float32(np.iinfo(np.int16).max)
                if np.issubdtype(self.dtype, np.signedinteger):
                    data = (self.data / scale).astype(dtype)
                else:
                    data = (self.data * scale).astype(dtype)

        if inplace:
            self.data = data
            return self
        else:
            return AudioChunk(
                file_path=self.file_path,
                begin=self.begin,
                end=self.end,
                sr=self.sr,
                data=data,
            )

    @staticmethod
    def _apply_fade(
        audio, sr: int, left_duration: float = 0.0, right_duration: float = 0.0
    ):
        # convert to audio indices (samples)
        l_fade_len = int(left_duration * sr)
        r_fade_len = int(right_duration * sr)
        r_end = audio.shape[0]
        r_start = r_end - r_fade_len

        # apply the curve
        if l_fade_len > 0:
            l_fade_curve = np.logspace(-1.0, 1.0, l_fade_len) ** 4.0 / 10000.0
            audio[0:l_fade_len] = audio[0:l_fade_len] * l_fade_curve
        if r_fade_len > 0:
            r_fade_curve = np.flip(np.logspace(-1.0, 1.0, r_fade_len) ** 4.0 / 10000.0)
            audio[r_start:r_end] = audio[r_start:r_end] * r_fade_curve

    @staticmethod
    def silence(duration: float, sr: int):
        return AudioChunk(
            data=np.zeros(
                int(duration * sr),
            ),
            sr=sr,
        )


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _flac_path = _wav_path.with_suffix(".flac")

    _audio_chunk = AudioChunk(_wav_path).load()
    _audio_chunk.save(_flac_path, overwrite=True)

    _flac_chunk = AudioChunk(_flac_path).load()

    for _audio_format in AudioFormat.names():
        _bytes = _audio_chunk.to_bytes(AudioFormat[_audio_format])
        _file_path = f"{_wav_path.stem}.{_audio_format}"
        with open(_file_path, "wb") as f:
            f.write(_bytes)
        print(_file_path, AudioChunk(_file_path).load().duration)
