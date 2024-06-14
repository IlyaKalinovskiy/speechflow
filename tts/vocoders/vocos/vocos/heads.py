from typing import Optional

import torch

from torch import nn
from torch.nn import functional as F
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.audio_codecs import (
    DAC,
)
from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from tts.acoustic_models.modules.common.blocks import Regression
from tts.vocoders.vocos.vocos.modules import symexp
from tts.vocoders.vocos.vocos.spectral_ops import IMDCT, ISTFT


class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class ISTFTHead(FourierHead):
    """ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".

    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        s = torch.polar(mag, p)
        audio = self.istft(s)
        return audio, None, {}


class IMDCTSymExpHead(FourierHead):
    """IMDCT Head module for predicting MDCT coefficients with symmetric exponential
    function.

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.

    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        sample_rate: Optional[int] = None,
        clip_audio: bool = False,
    ):
        super().__init__()
        out_dim = mdct_frame_len // 2
        self.out = nn.Linear(dim, out_dim)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)
        self.clip_audio = clip_audio

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.out.weight.mul_(scale.view(-1, 1))

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.out(x)
        x = symexp(x)
        x = torch.clip(
            x, min=-1e2, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio, None, {}


class IMDCTCosHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with parametrizing MDCT = exp(m) · cos(p)

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        mdct_frame_len: int,
        padding: str = "same",
        clip_audio: bool = False,
    ):
        super().__init__()
        self.clip_audio = clip_audio
        self.out = nn.Linear(dim, mdct_frame_len)
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)

    def forward(self, x: torch.Tensor, **kwargs):
        """Forward pass of the IMDCTCosHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.

        """
        x = self.out(x)
        m, p = x.chunk(2, dim=2)
        m = torch.exp(m).clip(
            max=1e2
        )  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(m * torch.cos(p))
        if self.clip_audio:
            audio = torch.clip(x, min=-1.0, max=1.0)

        return audio, None, {}


class DACHead(FourierHead):
    def __init__(
        self,
        dim: int,
        with_dac_loss: bool = False,
        dac_loss_every_iter: int = 1,
        dac_loss_max_iter: int = 1_000_000_000,
        with_sm_loss: bool = False,
        sm_loss_every_iter: int = 1,
        sm_loss_max_iter: int = 1_000_000_000,
        speaker_biometric_model: str = "speechbrain",
    ):
        super().__init__()
        self.with_dac_loss = with_dac_loss
        self.dac_loss_every_iter = dac_loss_every_iter
        self.dac_loss_max_iter = dac_loss_max_iter
        self.with_sm_loss = with_sm_loss
        self.sm_loss_every_iter = sm_loss_every_iter
        self.sm_loss_max_iter = sm_loss_max_iter

        self.dac_model = DAC()
        self.proj = Regression(dim, self.dac_model.embedding_dim, hidden_dim=dim)

        if with_sm_loss:
            self.bio = VoiceBiometricProcessor(speaker_biometric_model)
        else:
            self.bio = None

        self.register_buffer("current_iter", torch.LongTensor([0]))

    def forward(self, x, **kwargs):
        z_hat = self.proj(x)
        y_g_hat = self.dac_model.model.decoder(10.0 * z_hat.transpose(1, -1))

        losses = {}
        if (
            self.with_dac_loss
            and self.training
            and (self.current_iter + 1) % self.dac_loss_every_iter == 0
            and self.current_iter < self.dac_loss_max_iter
        ):
            if kwargs.get("ac_latent_gt", None) is not None:
                ac_latent_gt = kwargs["ac_latent_gt"]

                chunk = []
                for i, (a, b) in enumerate(kwargs.get("spec_chunk")):
                    chunk.append(ac_latent_gt[i, a:b, :])

                ac_latent_gt = torch.stack(chunk)
            else:
                with torch.no_grad():
                    ac_latent_gt = self.dac_model.encode(kwargs["audio_gt"].unsqueeze(1))
                    ac_latent_gt = ac_latent_gt.transpose(1, -1).detach()

            losses["dac_loss"] = F.mse_loss(z_hat, ac_latent_gt * 0.1)

        if (
            self.with_sm_loss
            and self.training
            and (self.current_iter + 1) % self.sm_loss_every_iter == 0
            and self.current_iter < self.sm_loss_max_iter
        ):
            audio_gt = kwargs.get("audio_gt").unsqueeze(1)
            losses["sm_loss"] = self.bio.compute_sm_loss(
                y_g_hat, audio_gt, sample_rate=self.dac_model.sample_rate
            )

        self.current_iter += 1
        return y_g_hat.squeeze(1), None, losses
