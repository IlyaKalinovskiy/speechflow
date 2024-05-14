import uuid
import random
import typing as tp
import logging
import tempfile
import subprocess as sp

from multiprocessing import current_process
from pathlib import Path

import numpy as np
import numpy.typing as npt

from scipy import signal
from torch.nn.functional import interpolate as torch_interpolate

from speechflow.data_pipeline.core.base_ds_processor import (
    BaseDSProcessor,
    ComputeBackend,
)
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing import (
    audio_codecs,
    ssl_models,
)
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.praat_sound_effects import (
    PraatSoundEffects,
)
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.data_pipeline.datasample_processors.tts_singletons import StatisticsRange
from speechflow.io import AudioChunk, Config
from speechflow.utils.fs import get_root_dir
from speechflow.utils.init import init_class_from_config, lazy_initialization

__all__ = [
    "SignalProcessor",
    "SSLProcessor",
    "ACProcessor",
    "monotonic_speech",
]

LOGGER = logging.getLogger("root")

try:
    import pyworld as pw
except ImportError as e:
    if current_process().name == "MainProcess":
        LOGGER.warning(f"pyworld is not available: {e}")

try:
    from torchaudio import functional as F
    from torchaudio import sox_effects, transforms
except ImportError as e:
    if current_process().name == "MainProcess":
        LOGGER.warning(f"torchaudio is not available: {e}")


class BaseAudioProcessor(BaseDSProcessor):
    def process(self, ds: tp.Union[AudioDataSample, tp.Any]):
        if ds.audio_chunk and not ds.audio_chunk.empty:
            assert np.issubdtype(
                ds.audio_chunk.waveform.dtype, np.floating
            ), "Audio data must be floating-point!"

        return super().process(ds)


class SignalProcessor(BaseAudioProcessor):
    def __init__(
        self,
        pipe: tp.Tuple[str, ...] = (),
        pipe_cfg: Config = Config.empty(),
        backend: ComputeBackend = ComputeBackend.librosa,
    ):
        super().__init__(pipe, pipe_cfg, backend)

        if "whisper" in pipe:
            self._sound_effects = init_class_from_config(
                PraatSoundEffects, pipe_cfg.section("whisper")
            )()
        else:
            self._sound_effects = None

    @PipeRegistry.registry(
        inputs={"file_path", "audio_chunk"},
        outputs={"audio_chunk"},
    )
    def process(self, ds: tp.Union[AudioDataSample, tp.Any]) -> AudioDataSample:
        return super().process(ds)

    @staticmethod
    def _quantize(s: npt.NDArray, bits: int):
        scale = np.float32(2**bits - 1)
        s = np.floor((s + 1.0) / 2.0 * scale + 0.5)
        return s.astype(np.int64)

    @staticmethod
    def _split_signal(s: npt.NDArray, bits: int):
        coarse = s // (2 ** (bits // 2))
        fine = s % (2 ** (bits // 2))
        return np.vstack([coarse, fine])

    @staticmethod
    def load(
        ds: AudioDataSample,
        sample_rate: tp.Optional[int] = None,
        dtype: npt.DTypeLike = np.float32,
        load_entire_file: bool = False,
    ) -> AudioDataSample:
        if ds.audio_chunk is None:
            if ds.file_path and ds.file_path.is_file():
                ds.audio_chunk = AudioChunk(file_path=ds.file_path)
            else:
                raise FileNotFoundError(f"File {ds.file_path.as_posix()} not found!")

            ds.audio_chunk.load(
                sr=sample_rate, dtype=dtype, load_entire_file=load_entire_file
            )
        else:
            if ds.audio_chunk.empty:
                ds.audio_chunk.load(
                    sr=sample_rate, dtype=dtype, load_entire_file=load_entire_file
                )
            else:
                if ds.audio_chunk.sr != sample_rate:
                    assert (
                        ds.audio_chunk.sr >= sample_rate
                    ), "target sample rate cannot be upper than current"
                    ds.audio_chunk.resample(sample_rate, inplace=True).astype(
                        dtype, inplace=True
                    )
        return ds

    @staticmethod
    def trim(
        ds: AudioDataSample,
        begin: tp.Optional[float] = None,
        end: tp.Optional[float] = None,
        min_duration: tp.Optional[float] = None,
        max_duration: tp.Optional[float] = None,
        random_chunk: bool = False,
        num_samples_per_chunk: tp.Optional[int] = None,
    ) -> AudioDataSample:
        def add_chunk_bound(_ds: AudioDataSample, _begin: float, _end: float):
            ds.additional_fields["audio_chunk"] = np.asarray((_begin, _end))
            _hop_len = ds.get_param_val("hop_len")
            if _hop_len:
                _s = 1 / _hop_len
                _len = round(_ds.audio_chunk.duration * ds.audio_chunk.sr * _s)
                ds.additional_fields["spec_chunk"] = np.asarray(
                    (int(_begin * _s), round(_end * _s))
                )
                assert _len == np.diff(ds.additional_fields["spec_chunk"])

        dura = ds.audio_chunk.duration

        if random_chunk and num_samples_per_chunk:
            y = ds.audio_chunk.waveform
            hop_len = ds.get_param_val("hop_len")
            assert y.size >= num_samples_per_chunk + 1
            begin = np.random.randint(low=0, high=y.size - num_samples_per_chunk + 1)
            begin = int(begin / hop_len) * hop_len
            y = y[begin : begin + num_samples_per_chunk]
            ds.audio_chunk = AudioChunk(data=y, sr=ds.audio_chunk.sr)
            add_chunk_bound(ds, begin, begin + num_samples_per_chunk)
            return ds

        if random_chunk:
            a = min_duration if min_duration else 0.1
            b = max_duration if max_duration else dura
            new_dura = a + (b - a) * random.random()
            begin = (dura - new_dura) * random.random()
            end = begin + new_dura
        else:
            if begin is None:
                begin = 0
            if end is None and max_duration is not None and dura > max_duration:
                end = max_duration

        ds.audio_chunk = ds.audio_chunk.trim(begin=begin, end=end)
        add_chunk_bound(ds, begin * ds.audio_chunk.sr, end * ds.audio_chunk.sr)

        if min_duration and ds.audio_chunk.duration < min_duration:
            raise RuntimeError("Invalid wave duration.")

        if max_duration and ds.audio_chunk.duration > max_duration:
            raise RuntimeError("Invalid wave duration.")

        return ds

    @staticmethod
    def pad(ds: AudioDataSample, pad_size: float = 0.25) -> AudioDataSample:
        data = ds.audio_chunk.waveform
        n = int(pad_size * ds.audio_chunk.sr)
        data = np.pad(data, (n, n), "constant", constant_values=(0, 0))
        ds.audio_chunk.data = data
        ds.audio_chunk.end += 2.0 * pad_size
        return ds

    @staticmethod
    def astype(ds: AudioDataSample, dtype=np.float32) -> AudioDataSample:
        ds.audio_chunk.astype(dtype, inplace=True)
        return ds

    @staticmethod
    def resample(ds: AudioDataSample, sample_rate: int) -> AudioDataSample:
        ds.audio_chunk.resample(sample_rate, inplace=True)
        return ds

    @staticmethod
    def preemphasis(ds: AudioDataSample, beta: float = 0.97) -> AudioDataSample:
        waveform, f32 = ds.audio_chunk.waveform, np.float32
        assert np.issubdtype(
            waveform.dtype, np.floating
        ), "Audio data must be floating-point!"
        waveform = signal.lfilter([f32(1), -f32(beta)], [f32(1)], waveform)
        ds.audio_chunk.data = waveform
        return ds

    @staticmethod
    def inv_preemphasis(ds: AudioDataSample, beta: float = 0.97) -> AudioDataSample:
        waveform, f32 = ds.audio_chunk.waveform, np.float32
        waveform = signal.lfilter([f32(1)], [f32(1), -f32(beta)], waveform)
        ds.audio_chunk.data = waveform
        return ds

    @staticmethod
    def mu_law_encode(
        ds: AudioDataSample,
        bits: int = 9,
        quantize: bool = False,
        split: bool = False,
    ):
        waveform = ds.audio_chunk.waveform
        assert np.issubdtype(
            waveform.dtype, np.floating
        ), "Audio data must be floating-point!"

        s = waveform

        if bits < 16:
            mu = np.float32(2**bits - 1)
            s = np.sign(waveform) * np.log(1.0 + mu * np.abs(waveform)) / np.log(1.0 + mu)

        if quantize:
            s = SignalProcessor._quantize(s, bits)

        if split:
            assert quantize
            s = SignalProcessor._split_signal(s, bits)

        ds.mu_law_waveform = s
        ds.bits = bits
        return ds

    @staticmethod
    def mu_law_decode(ds: AudioDataSample):
        mu_law = ds.mu_law_waveform
        bits = ds.bits
        n_classes = 2 ** (bits // 2)

        if mu_law.ndim == 2:
            mu_law = mu_law[0, :] * n_classes + mu_law[1, :]
        elif mu_law.ndim == 3:
            mu_law = mu_law[:, 0, :] * n_classes + mu_law[:, 1, :]

        mu = np.float32(2**bits - 1)
        s = mu_law.astype(np.float32)

        if np.issubdtype(mu_law.dtype, np.int64):
            s = 2.0 * (s / mu) - 1.0

        if bits < 16:
            s = np.sign(s) / mu * ((1.0 + mu) ** np.abs(s) - 1.0)

        ds.audio_chunk.data = s
        return ds

    @staticmethod
    def add_noise(ds: AudioDataSample):
        noise = np.random.randn(*ds.audio_chunk.data.shape).astype(np.float32)
        if np.issubdtype(ds.audio_chunk.dtype, np.floating):
            scale = np.float32(np.iinfo(np.int16).max)
            noise *= 1 / scale
        else:
            noise = noise.astype(np.int16)
        ds.audio_chunk.data += noise
        return ds

    def whisper(self, ds: AudioDataSample, p: float = 1.0):
        if random.random() > p:
            return ds

        ds.audio_chunk = self._sound_effects.whisper(ds.audio_chunk)
        return ds

    @staticmethod
    def ffmpeg_loudnorm(ds: AudioDataSample):
        duration = ds.audio_chunk.duration
        sample_rate = ds.audio_chunk.sr
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                wav_path_in = Path(tmp_dir) / f"{uuid.uuid4()}.wav"
                wav_path_out = Path(tmp_dir) / f"{uuid.uuid4()}.wav"
                ds.audio_chunk.save(wav_path_in)

                cmd_base = "ffmpeg -y -i " + wav_path_in.as_posix() + " -af "
                if duration < 3:
                    cmd_base += "apad,atrim=0:3,"
                cmd = (
                    cmd_base
                    + f"loudnorm=I=-16:dual_mono=true:TP=-1.5:LRA=11:print_format=summary -ar {sample_rate} -f null -"
                )
                output = sp.run(cmd.split(), capture_output=True)
                for line in output.stderr.decode().split("\n"):
                    if "Input Integrated" in line:
                        input_i = float(
                            line[len("Input Integrated:") :].strip().split()[0]
                        )
                    elif "Input True Peak" in line:
                        input_tp = float(line[len("Input True Peak:") :].split()[0])
                    elif "Input LRA" in line:
                        input_lra = float(line[len("Input LRA:") :].split()[0])
                    elif "Input Threshold" in line:
                        input_thresh = float(line[len("Input Threshold:") :].split()[0])
                    elif "Target Offset" in line:
                        target_offset = float(line[len("Target Offset:") :].split()[0])
                cmd = (
                    cmd_base
                    + f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:measured_thresh={input_thresh}:offset={target_offset}:linear=true:print_format=summary"
                )
                if duration < 3:
                    cmd += f",atrim=0:{duration} "
                cmd += f" -ar {sample_rate} {wav_path_out.as_posix()}"
                sp.run(cmd.split(), capture_output=True)

                ds.audio_chunk.waveform = AudioChunk(wav_path_out).load().waveform
            except Exception as e:
                LOGGER.error(e)

        return ds


class SSLProcessor(BaseAudioProcessor):
    def __init__(
        self,
        ssl_type: str,
        ssl_params: Config = Config.empty(),
        resize_from: tp.Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self._ssl_cls = getattr(ssl_models, ssl_type)
        self._ssl_params = ssl_params
        self._resize_from = resize_from
        self._ssl_model = None

    def init(self):
        super().init()
        self._ssl_params["device"] = self.device
        self._ssl_model = init_class_from_config(
            self._ssl_cls, self._ssl_params, check_keys=False
        )()

    @PipeRegistry.registry(
        inputs={"audio_chunk"},
        outputs={"ssl_feat"},
    )
    @lazy_initialization
    def process(self, ds: tp.Union[AudioDataSample, tp.Any]) -> AudioDataSample:
        ds = super().process(ds)
        assert self._ssl_model is not None

        ds.ssl_feat = self._ssl_model(ds.audio_chunk)

        if self._resize_from:
            attr = getattr(ds, self._resize_from)
            scale = (attr.shape[0] + 1) / ds.ssl_feat.encode.shape[0]
            feat = ds.ssl_feat.encode.t().unsqueeze(0)
            ds.ssl_feat.encode = torch_interpolate(feat, scale_factor=scale)
            ds.ssl_feat.encode = ds.ssl_feat.encode.squeeze(0).t()
            ds.ssl_feat.encode = ds.ssl_feat.encode[: attr.shape[0]]
            assert (
                ds.ssl_feat.encode.shape[0] == attr.shape[0]
            ), f"shape mismatch {ds.ssl_feat.encode.shape[0]} != {attr.shape[0]}"

        return ds.to_numpy()


class ACProcessor(BaseAudioProcessor):
    def __init__(
        self,
        ac_type: str,
        ac_params: Config = Config.empty(),
        resynt: bool = False,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self._ac_cls = getattr(audio_codecs, ac_type)
        self._ac_params = ac_params
        self._resynt = resynt
        self._ac_model = None

    def init(self):
        super().init()
        self._ac_params["device"] = self.device
        self._ac_model = init_class_from_config(
            self._ac_cls, self._ac_params, check_keys=False
        )()

    @PipeRegistry.registry(
        inputs={"audio_chunk"},
        outputs={"ac_feat"},
    )
    @lazy_initialization
    def process(self, ds: tp.Union[AudioDataSample, tp.Any]) -> AudioDataSample:
        ds = super().process(ds)
        assert self._ac_model is not None

        ds.ac_feat = self._ac_model(ds.audio_chunk, ds=ds, vq_only=not self._resynt)

        if self._resynt:
            if ds.ac_feat.waveform is None:
                feat = ds.ac_feat.encode.t().unsqueeze(0)
                waveform = self._ac_model.decode(feat.to(self._ac_model.device))
                waveform = waveform.squeeze().cpu().numpy()
            else:
                waveform = ds.ac_feat.waveform.numpy()

            pad = len(ds.audio_chunk.waveform) - len(waveform)
            if pad > 0:
                waveform = np.pad(waveform, (0, pad), mode="symmetric")
            elif pad < 0:
                waveform = waveform[:pad]

            ds.audio_chunk = AudioChunk(data=waveform, sr=self._ac_model.sample_rate)

        return ds.to_numpy()


@PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"audio_chunk"})
def monotonic_speech(
    ds: AudioDataSample, ranges: StatisticsRange = None
) -> AudioDataSample:
    x = ds.audio_chunk.waveform.astype(np.float64)
    _f0, t = pw.dio(x, ds.audio_chunk.sr)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, ds.audio_chunk.sr)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, ds.audio_chunk.sr)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, ds.audio_chunk.sr)  # extract aperiodicity

    if ranges is not None:
        f0_mean, _ = ranges.get_stat("pitch", ds.speaker_name)
    else:
        f0_mean = np.mean(f0)

    # smooth pitch to synthesize monotonic speech
    v = f0 > 0
    uv = f0 <= 0
    if any(v):
        f0 = np.ones_like(f0) * f0_mean
        f0[uv] = 0

    y = pw.synthesize(
        f0, sp, ap, ds.audio_chunk.sr
    )  # synthesize an utterance using the parameters
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))

    if not np.isfinite(y).all():
        return ds

    assert len(y) >= len(x)
    ds.audio_chunk.waveform = y[: len(x)].astype(np.float32)
    ds.transform_params["monotonic_speech"] = {}
    return ds


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    matplotlib.use("TkAgg")

    _file_path = get_root_dir() / "tests/data/test_audio.wav"

    _cfg: Config = Config(
        {
            "load": {"sample_rate": 22050},
            "trim": {"end": 3},
            "ssl_features": {
                "ssl_type": "Wav2Vec",
                "device": "cpu",
            },
            "ac_features": {
                "ac_type": "DAC",
                "device": "cpu",
            },
        }
    )

    signal_proc = SignalProcessor(("load", "trim"), _cfg)
    ssl_proc = SSLProcessor(**_cfg["ssl_features"])
    ac_proc = ACProcessor(**_cfg["ac_features"])

    _ds = AudioDataSample(file_path=_file_path)
    _ds = signal_proc.process(_ds)
    _ds = ssl_proc.process(_ds)
    _ds = ac_proc.process(_ds)

    for name in ["ssl_feat", "ac_feat"]:
        _data = getattr(_ds, name)
        fig, ax = plt.subplots(1, 1, dpi=160, facecolor="w")  # type: ignore
        im = ax.imshow(np.flip(_data[:], axis=1).T)
        divider = make_axes_locatable(ax)
        fig.colorbar(im, cax=divider.append_axes("right", size=0.25, pad=0.05))
        ax.set_title(name)

    plt.show()
