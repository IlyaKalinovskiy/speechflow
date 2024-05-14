import random
import typing as tp
import logging

from functools import wraps
from multiprocessing import current_process
from pathlib import Path

import numpy as np
import torch
import librosa

from librosa.core import istft, stft
from librosa.util import fix_length
from psola import vocode
from scipy.signal import butter, resample, sosfilt

from speechflow.data_pipeline.core.base_ds_processor import (
    BaseDSProcessor,
    ComputeBackend,
)
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.io import AudioChunk, Config
from speechflow.utils.fs import get_root_dir

__all__ = [
    "WaveAugProcessor",
    "SpecAugProcessor",
]

LOGGER = logging.getLogger("root")

try:
    from torchaudio import functional as F
    from torchaudio import sox_effects, transforms
except ImportError as e:
    if current_process().name == "MainProcess":
        LOGGER.warning(f"torchaudio is not available: {e}")


def check_probability(method: tp.Callable) -> tp.Callable:
    """Decorator for applying augmentation with probability.

    Probability must be in range [0, 1].

    """

    @wraps(method)
    def use_augmentation(*args, **kwargs) -> AudioDataSample:
        p = kwargs.get("p", 1.0)
        if (p < 0) or (p > 1):
            raise ValueError(f"probability of applying aug p must be in [0, 1]. Got {p}.")

        if random.random() > p:
            return kwargs["ds"]

        if len(args) == 0:
            return method(**kwargs)
        else:
            return method(args[0], **kwargs)

    return use_augmentation


class WaveAugProcessor(BaseDSProcessor):
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.librosa,
        shuffle: bool = False,
        p: float = 1.0,
    ):
        super().__init__(pipe, pipe_cfg, backend)
        self.shuffle = shuffle
        self.p = p

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"audio_chunk"})
    def process(self, ds: AudioDataSample) -> AudioDataSample:
        if ds.audio_chunk and not ds.audio_chunk.empty:
            assert np.issubdtype(
                ds.audio_chunk.waveform.dtype, np.floating
            ), "Audio data must be floating-point!"

        ds.transform_params.update(self.transform_params)

        if random.random() > self.p:
            return ds

        handlers = list(self.components.values())
        if self.shuffle:
            random.shuffle(handlers)

        tmp_audio_chunk = ds.audio_chunk.copy()
        for handler in handlers:
            if hasattr(handler, "keywords"):
                p = handler.keywords.get("p", 1.0)  # type: ignore
                ds = handler(ds=ds, p=p)
            else:
                raise NotImplementedError()

        if not np.isfinite(ds.audio_chunk.waveform).all():
            ds.audio_chunk = tmp_audio_chunk

        return ds

    @staticmethod
    def _get_random_curve(
        size: int,
        min_points: int,
        max_points: int,
        min_ratio: float,
        max_ratio: float,
    ) -> np.ndarray:
        num_points = random.randint(min_points, max_points)
        curve = np.random.uniform(min_ratio, max_ratio, (num_points,))
        curve = resample(curve, size)
        return curve

    @staticmethod
    def _butter_bandstop_filter(
        data: np.ndarray, lowcut: int, highcut: int, sample_rate: int
    ):
        data_dtype = data.dtype
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(5, [low, high], btype="bandstop", output="sos")
        y = sosfilt(sos, data).astype(data_dtype)
        return y

    @staticmethod
    def _calculate_desired_noise_rms(clean_rms: float, snr: float):
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a)
        return noise_rms

    @staticmethod
    def _get_random_slice(waveform: np.ndarray, slice_size: int):
        if len(waveform) < slice_size:
            waveform = np.pad(waveform, (0, slice_size - len(waveform)), "constant")
        else:
            offset = random.randint(0, len(waveform) - slice_size)
            waveform = waveform[offset : offset + slice_size]
        return waveform

    @staticmethod
    @check_probability
    def random_segment(
        ds: AudioDataSample,
        min_size: int,
        max_size: int,
        p: float = 1.0,
    ) -> AudioDataSample:
        """Cut random segment from mu_law.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_size (float): Min size of segment (in seconds)
            max_size (float): Max size of segment (in seconds)

        Returns: ds (WaveDataSample)

        """

        segment_size = random.randint(
            min(ds.mu_law_waveform.shape[0], min_size),
            min(ds.mu_law_waveform.shape[0], max_size),
        )
        offset = random.randint(0, ds.mu_law_waveform.shape[0] - segment_size)

        ds.mu_law_waveform = ds.mu_law_waveform[offset : segment_size + offset]

        return ds

    @check_probability
    def pitch_shift(
        self,
        ds: AudioDataSample,
        min_semitones: int = -4,
        max_semitones: int = 4,
        p: float = 1.0,
    ) -> AudioDataSample:
        """Change pitch without changing the tempo.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_semitones (int): Min semitones, must be in range [-13, max_semitones)
            max_semitones (int): Max semitones, must be in range (min_semitones, 13]

        Returns: ds (WaveDataSample)

        """

        if max_semitones > 13 or min_semitones < -13:
            raise ValueError(
                f"abs. value of max_semitones and min_semitones must be lower than 13, "
                f"got {max_semitones} and {min_semitones}"
            )

        num_semitones = random.uniform(min_semitones, max_semitones)

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.audio_chunk.data = librosa.effects.pitch_shift(
                ds.audio_chunk.data, sr=ds.audio_chunk.sr, n_steps=num_semitones
            )
        else:
            raise NotImplementedError(
                f"pitch_shift is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def time_stretch(
        self,
        ds: AudioDataSample,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
        p: float = 1.0,
    ):
        """Change tempo without changing the pitch.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_rate (float): Min stretch rate
            max_rate (float): Max stretch rate

        Returns: ds (WaveDataSample)

        """

        rate = random.uniform(min_rate, max_rate)

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.audio_chunk.data = librosa.effects.time_stretch(
                ds.audio_chunk.data, rate=rate
            )
        else:
            raise NotImplementedError(
                f"time_stretch is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def gain(
        self,
        ds: AudioDataSample,
        min_ratio: float = 0.5,
        max_ratio: float = 2.0,
        p: float = 1.0,
    ) -> AudioDataSample:
        randomized_gain = np.random.rand() * (max_ratio - min_ratio) + min_ratio

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.audio_chunk.data = np.clip(ds.audio_chunk.data * randomized_gain, -1, 1)
        else:
            raise NotImplementedError(
                f"gain_curve is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def gain_curve(
        self,
        ds: AudioDataSample,
        min_points: int = 2,
        max_points: int = 5,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
        p: float = 1.0,
    ) -> AudioDataSample:
        """Multiply audio by random gain curve.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_points (int): Min points in curve
            max_points (int): Max points in curve
            min_ratio (float): Min value in each point before interpolation
            max_ratio (float): Max value in each point before interpolation

        Returns: ds (WaveDataSample)

        """

        randomized_curve = self._get_random_curve(
            min_points=min_points,
            max_points=max_points,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            size=len(ds.audio_chunk.data),
        ).astype(np.float32)

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.audio_chunk.data = np.clip(ds.audio_chunk.data * randomized_curve, -1, 1)
        else:
            raise NotImplementedError(
                f"gain_curve is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def clipping_distortion(
        self,
        ds: AudioDataSample,
        min_percentile_threshold: int = 5,
        max_percentile_threshold: int = 15,
        p: float = 1.0,
    ) -> AudioDataSample:
        """Clip random percentile of audio.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_percentile_threshold (int): Min percentile
            max_percentile_threshold (int): Max percentile

        Returns: ds (WaveDataSample)

        """

        if min_percentile_threshold > max_percentile_threshold:
            raise ValueError(
                f"max_percentile threshold must be greater than min_percentile_threshold. "
                f"Got {min_percentile_threshold} and {max_percentile_threshold}"
            )

        lower_percentile_threshold = random.randint(
            min_percentile_threshold, max_percentile_threshold
        )

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            lower_threshold, upper_threshold = np.percentile(
                ds.audio_chunk.data,
                [lower_percentile_threshold, 100 - lower_percentile_threshold],
            )
            ds.audio_chunk.data = np.clip(
                ds.audio_chunk.data, lower_threshold, upper_threshold
            )

        else:
            raise NotImplementedError(
                f"clipping_distortion is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def frequency_mask(
        self,
        ds: AudioDataSample,
        min_frequency_band: float = 0.0,
        max_frequency_band: float = 0.5,
        p: float = 1.0,
    ) -> AudioDataSample:
        """Mask some frequency band on the spectrogram.

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform
            min_frequency_band (float): Min size of frequency band
            max_frequency_band (float): Max size of frequency band

        Returns: ds (WaveDataSample)

        """

        if min_frequency_band > max_frequency_band:
            raise ValueError(
                f"min_frequency_band threshold must be greater than max_frequency_band. "
                f"Got {min_frequency_band} and {max_frequency_band}"
            )

        bandwidth = random.randint(
            int((min_frequency_band * ds.audio_chunk.sr) // 2),
            int((max_frequency_band * ds.audio_chunk.sr) // 2),
        )

        freq_start = random.randint(16, int(ds.audio_chunk.sr // 2 - bandwidth - 1))

        if self.backend in [
            ComputeBackend.librosa,
            ComputeBackend.torchaudio,
            ComputeBackend.nvidia,
        ]:
            ds.audio_chunk.data = self._butter_bandstop_filter(
                ds.audio_chunk.data, freq_start, freq_start + bandwidth, ds.audio_chunk.sr
            )

        else:
            raise NotImplementedError(
                f"time_stretch is not implemented for {self.backend} backend."
            )

        return ds

    @check_probability
    def gsm_simulation(self, ds: AudioDataSample, *kwargs) -> AudioDataSample:
        """

        Args:
            ds (AudioDataSample): Datasample with mu_law waveform

        Returns: ds (WaveDataSample)

        """

        if self.backend in [ComputeBackend.torchaudio]:
            waveform, sample_rate = sox_effects.apply_effects_tensor(
                torch.from_numpy(ds.audio_chunk.data).unsqueeze(0).to(torch.float32),
                ds.audio_chunk.sr,
                effects=[
                    ["lowpass", "4000"],
                    [
                        "compand",
                        "0.02,0.05",
                        "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                        "-8",
                        "-7",
                        "0.05",
                    ],
                    ["rate", "8000"],
                ],
            )
            signal_gsm = F.apply_codec(waveform, sample_rate, format="gsm")
            signal_gsm = F.resample(signal_gsm, sample_rate, ds.audio_chunk.sr)

            ds.audio_chunk.data = signal_gsm.numpy().squeeze()
        else:
            raise NotImplementedError(
                f"time_stretch is not implemented for {self.backend} backend."
            )
        return ds

    @check_probability
    def change_rhythm(
        self,
        ds: AudioDataSample,
        mode: str = "up",
        seg_size: float = 0.16,
        silent_front: float = 0.48,
        silent_end: float = 0.32,
        p: float = 1.0,
    ) -> AudioDataSample:
        def gen_curve(
            n_segments,
            mode: str = "fsf",
            max: float = 1.2,
            min: float = 0.8,
            const: float = 1.0,
        ):
            rates = [0.0] * n_segments

            if mode == "constant":
                rates = [const] * n_segments
            elif mode == "fsf":  # fast-slow-fast(0.5-2-0.5)
                split = int(n_segments / 3)
                for i in range(split):
                    rates[i] = max
                for i in range(split, split * 2):
                    rates[i] = min
                for i in range(split * 2, n_segments):
                    rates[i] = max
            elif mode == "parabola":
                x = np.array(range(n_segments))
                a = 4 * (min - max) / (n_segments * n_segments)
                rates = a * (x - n_segments / 2) ** 2 + max
            elif mode == "down":
                x = np.array(range(n_segments))
                rates = (min - max) / n_segments * x + max
            elif mode == "up":
                x = np.array(range(n_segments))
                rates = (max - min) / n_segments * x + min
            elif mode == "question":
                k = 4 * (max - 1) / n_segments
                for x in range(int(n_segments * 0.75), n_segments):
                    rates[x] = max(1.0, k * x - 3 * max + 4)
            elif mode == "stress":
                k = 4 * (1 - max) / n_segments
                for x in range(int(n_segments * 0.5), int(n_segments * 0.75)):
                    rates[x] = k * x + 3 * max - 2
            else:
                raise NotImplementedError
            return rates

        audio = ds.audio_chunk.waveform
        sr = ds.audio_chunk.sr
        seg_size = int(seg_size * sr)
        silent_front = int(silent_front / seg_size)
        silent_end = int(silent_end / seg_size)
        N = len(audio)

        if N % seg_size != 0:
            padding = int((N // seg_size + 1) * seg_size - N)
            audio = np.append(audio, [0.0] * padding)
            N = len(audio)
        assert N % seg_size == 0
        n_segments = int(N // seg_size - silent_front - silent_end)

        rates = (
            [1.0] * silent_front + list(gen_curve(n_segments, mode)) + [1.0] * silent_end
        )

        output_audio = []
        for i in range(n_segments):
            segment = audio[i * seg_size : (i + 1) * seg_size]
            output_audio.append(
                vocode(audio=segment, sample_rate=sr, constant_stretch=rates[i])
            )

        output_audio = np.hstack(output_audio)
        ds.audio_chunk = AudioChunk(data=output_audio, sr=sr)
        return ds

    @check_probability
    def vtlp(
        self,
        ds: AudioDataSample,
        alpha_min: float = 0.9,
        alpha_max: float = 1.1,
        p: float = 1.0,
    ) -> AudioDataSample:
        def warp_freq(n_fft, fs, fhi=4800, alpha=0.9):
            bins = np.linspace(0, 1, n_fft)
            f_warps = []

            scale = fhi * min(alpha, 1)
            f_boundary = scale / alpha
            fs_half = fs // 2

            for k in bins:
                f_ori = k * fs
                if f_ori <= f_boundary:
                    f_warp = f_ori * alpha
                else:
                    f_warp = fs_half - (fs_half - scale) / (fs_half - scale / alpha) * (
                        fs_half - f_ori
                    )
                f_warps.append(f_warp)

            return np.array(f_warps)

        x = ds.audio_chunk.waveform  # monotonic wave
        alpha = np.random.uniform(low=alpha_min, high=alpha_max)

        fs = ds.audio_chunk.sr
        S = stft(x).T
        T, K = S.shape
        dtype = S.dtype

        f_warps = warp_freq(K, fs, alpha=alpha)
        f_warps *= (K - 1) / max(f_warps)
        new_S = np.zeros([T, K], dtype=dtype)

        for k in range(K):
            # first and last freq
            if k == 0 or k == K - 1:
                new_S[:, k] += S[:, k]
            else:
                warp_up = f_warps[k] - np.floor(f_warps[k])
                warp_down = 1 - warp_up
                pos = int(np.floor(f_warps[k]))

                new_S[:, pos] += warp_down * S[:, k]
                new_S[:, pos + 1] += warp_up * S[:, k]

        y = istft(new_S.T)
        y = fix_length(y, size=len(x))

        ds.audio_chunk.waveform = y.astype(np.float32)
        return ds


class SpecAugProcessor(WaveAugProcessor):
    @PipeRegistry.registry(inputs={"magnitude", "mel"}, outputs={"magnitude", "mel"})
    def process(self, ds: AudioDataSample) -> AudioDataSample:
        ds.transform_params.update(self.transform_params)

        if random.random() > self.p:
            return ds

        handlers = list(self.components.values())
        if self.shuffle:
            random.shuffle(handlers)

        for handler in handlers:
            if hasattr(handler, "keywords"):
                p = handler.keywords.get("p", 1.0)  # type: ignore
                ds = handler(ds=ds, p=p)
            else:
                raise NotImplementedError()

        return ds

    @check_probability
    def blur(
        self, ds: SpectrogramDataSample, radius: int = (1, 3, 5), sigma=None, *kwargs
    ):
        from scipy.ndimage.filters import gaussian_filter

        if sigma is None:
            sigma = 0.75 * random.random()

        if isinstance(radius, tp.Iterable):
            r = np.random.choice(radius)
        else:
            r = radius

        ds.mel = gaussian_filter(ds.mel, sigma=sigma, radius=r)
        return ds

    @check_probability
    def noise(self, ds: SpectrogramDataSample, var: float = 1.0, scale=None, *kwargs):
        if scale is None:
            scale = 0.2 * random.random()

        mel_noise = ds.mel + scale * np.random.normal(0, var, ds.mel.shape)
        ds.mel = mel_noise.astype(np.float32)
        return ds


if __name__ == "__main__":

    from speechflow.data_pipeline.datasample_processors.audio_processors import (
        SignalProcessor,
    )

    wav_path = get_root_dir() / "tests/data/test_audio.wav"

    _pipe_cfg = Config(
        {
            "load": {"sample_rate": 22050},
            "gain_curve": {
                "p": 0.5,
                "min_ratio": 0.0,
                "max_ratio": 2.0,
            },
            "gsm_simulation": {"p": 0.5},
        }
    )

    signal_proc = SignalProcessor(("load",), _pipe_cfg)

    _pipe = (
        "gain_curve",
        "pitch_shift",
        "change_rhythm",
        "vtlp",
        # "gsm_simulation",
    )

    temp_folder = Path("temp")
    temp_folder.mkdir(exist_ok=True)

    for i in range(10):
        wave_aug = WaveAugProcessor(_pipe, _pipe_cfg, shuffle=True)

        _ds = AudioDataSample(file_path=wav_path)
        _ds = signal_proc.process(_ds)
        _ds = wave_aug.process(_ds)

        _ds.audio_chunk.save(temp_folder / f"test_aug_test_{i}.wav", overwrite=True)
