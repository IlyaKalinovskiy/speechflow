import typing as tp

import torch

from torch.nn import functional as F

from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.audio_codecs import (
    DAC,
)
from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from speechflow.io import check_path, tp_PATH
from tts.acoustic_models.modules.common.blocks import Regression
from tts.vocoders.vocos.modules.heads.fourier import FourierHead

__all__ = ["DACHead"]


class DACHead(FourierHead):
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        dim: int,
        with_dac_loss: bool = False,
        dac_loss_every_iter: int = 1,
        dac_loss_max_iter: int = 1_000_000_000,
        with_sm_loss: bool = False,
        sm_loss_every_iter: int = 1,
        sm_loss_max_iter: int = 1_000_000_000,
        speaker_biometric_model: tp.Literal["speechbrain", "wespeaker"] = "speechbrain",
        pretrain_path: tp.Optional[tp_PATH] = None,
    ):
        super().__init__()
        self.with_dac_loss = with_dac_loss
        self.dac_loss_every_iter = dac_loss_every_iter
        self.dac_loss_max_iter = dac_loss_max_iter
        self.with_sm_loss = with_sm_loss
        self.sm_loss_every_iter = sm_loss_every_iter
        self.sm_loss_max_iter = sm_loss_max_iter

        self.dac_model = DAC(pretrain_path=pretrain_path)
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
            self.training
            and self.with_dac_loss
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
            self.training
            and self.with_sm_loss
            and (self.current_iter + 1) % self.sm_loss_every_iter == 0
            and self.current_iter < self.sm_loss_max_iter
        ):
            audio_gt = kwargs.get("audio_gt").unsqueeze(1)
            losses["sm_loss"] = self.bio.compute_sm_loss(
                y_g_hat, audio_gt, sample_rate=self.dac_model.sample_rate
            )

        self.current_iter += 1
        return y_g_hat.squeeze(1), None, losses
