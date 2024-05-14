import abc
import typing as tp

import torch
import torch.nn as nn

from tts.vocoders.vocos.common.pqmf import PQMF


class BaseTransform(nn.Module, abc.ABC):
    """Base Waveform Transform."""

    @abc.abstractmethod
    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        pass

    def forward(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        return self.transform(waveform, global_step)


class TransformComposition(BaseTransform):
    def __init__(self, transform_configuration: tp.Union[tp.List[tp.Dict], tp.Dict]):
        super().__init__()

        if type(transform_configuration) == dict:
            transform_configuration = [transform_configuration]

        transforms_composition = []
        for transform_config in transform_configuration:
            cls = globals()[transform_config["class_name"]]
            transform = cls(**transform_config.get("initialization_params", {}))
            transforms_composition.append(transform)
        self.transforms_composition = nn.ModuleList(*transforms_composition)

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        for transform in self.transforms_composition:
            waveform = transform(waveform, global_step)
        return waveform


class IdentityTransform(BaseTransform):
    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        return waveform


class SpectrogramTransform(BaseTransform):
    """Simple transform for computing STFT from given waveform.

    Args:
        fft_size (int): fft_size for stft
        hop_size (int): hop_size for stft
        win_size (int): win_size for stft

    """

    def __init__(self, fft_size: int = 1024, hop_size: int = 256, win_size: int = 800):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.window = nn.Parameter(torch.hann_window(win_size), requires_grad=False)

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(1)

        # Always compute spectrogram in FP32:
        input_tensor_dtype = waveform.type()
        if input_tensor_dtype != torch.float32:
            waveform = waveform.type(torch.float32)
            self.window.data = self.window.data.type(torch.float32)

        x_stft = torch.stft(
            waveform,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.data,
            return_complex=False,
        )
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        outputs = torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
        outputs = torch.sqrt(outputs)

        if input_tensor_dtype != torch.float32:
            outputs = outputs.type(input_tensor_dtype)

        return outputs.unsqueeze(1)


class DownsampleTransform(BaseTransform):
    """Simple transform for down-sampling input signal.

    Args:
        factor (int): ratio for down-sampling.
        use_dwt (bool): Whether to use Discrete Wavelet Transform to downsample signal.

    """

    def __init__(self, factor: int = 1):
        super().__init__()

        self._factor = factor
        if factor == 1:
            self.downsample_module = nn.Identity()
        else:
            self.downsample_module = nn.AvgPool1d(
                kernel_size=factor * 2,
                stride=factor,
                padding=(factor * 2 - 1) // 2,
                count_include_pad=False,
            )

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        return self.downsample_module(waveform)


class PQMFTransform(BaseTransform, PQMF):  # type: ignore
    """
    Args:
        pipe (List[str]): pipe for transform (['analysis'], ['synthesis'], ['analysis', 'synthesis']
        subbands (int): The number of subbands.
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    """

    def __init__(
        self,
        pipe: tp.List[str],
        subbands: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ):
        super().__init__(subbands, taps, cutoff_ratio, beta)
        self.pipe = pipe

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        for function_name in self.pipe:
            waveform = super().__getattribute__(function_name)(waveform)
        return waveform


class LinearNoiseSchedulerTransform(BaseTransform):
    def __init__(
        self,
        max_weight: float = 0.4,
        min_weight: float = 0.3,
        min_iter: int = 0,
        max_iter: int = 10000,
    ):
        super().__init__()
        self.iteration = 0
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_iter = min_iter
        self.max_iter = max_iter

    def _compute_weight(self, global_step: int) -> float:
        weight = max(
            (
                1
                - min(
                    max((global_step - self.min_iter), 0)
                    / (self.max_iter - self.min_iter),
                    1,
                )
            )
            * self.max_weight,
            self.min_weight,
        )
        return weight

    def transform(self, waveform: torch.Tensor, global_step: int) -> torch.Tensor:
        w = self._compute_weight(global_step=global_step)
        return waveform + torch.randn_like(waveform) * w
