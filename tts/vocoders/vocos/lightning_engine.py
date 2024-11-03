import io
import math
import typing as tp

import numpy as np
import torch
import torchaudio
import transformers
import pytorch_lightning as pl

from clearml import Task

from speechflow.io import AudioChunk, tp_PATH
from speechflow.training.saver import ExperimentSaver
from tts.vocoders.batch_processor import VocoderBatchProcessor
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos import losses
from tts.vocoders.vocos.helpers import plot_spectrogram_to_numpy
from tts.vocoders.vocos.modules.backbones.base import Backbone
from tts.vocoders.vocos.modules.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor
from tts.vocoders.vocos.modules.heads.base import WaveformGenerator
from tts.vocoders.vocos.utils.tensor_utils import safe_log

__all__ = ["VocosLightningEngine"]


class VocosLightningEngine(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: WaveformGenerator,
        batch_processor: VocoderBatchProcessor,
        saver: ExperimentSaver,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 1.0,
        mrd_loss_coeff: float = 1.0,
        auxiliary_loss_coeff: float = 1.0,
        auxiliary_losses_every: int = 1,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        use_sm_loss: bool = False,
        use_wavlm_loss: bool = False,
        use_cdpam_loss: bool = False,
        biometric_model_type: tp.Literal["speechbrain", "wespeaker"] = "wespeaker",
        biometric_model_name: tp.Optional[tp_PATH] = None,
        wavlm_model_name: str = "microsoft/wavlm-base-plus",
        loss_device: str = "cpu",
        fft_sizes: tp.Tuple[int, ...] = (1024, 680, 450),
        hop_sizes: tp.Tuple[int, ...] = (200, 135, 90),
        win_sizes: tp.Tuple[int, ...] = (800, 450, 300),
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        use_clearml_logger: bool = False,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (BaseHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            batch_processor (VocoderBatchProcessor): An instance of VocoderBatchProcessor for convert batch to model inputs.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()

        assert pl.__version__ == "1.8.6", RuntimeError(
            "pytorch_lightning==1.8.6 required"
        )

        self.save_hyperparameters(
            ignore=["feature_extractor", "backbone", "head", "batch_processor", "saver"]
        )

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self.batch_processor = batch_processor
        self.saver = saver

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = losses.DiscriminatorLoss()
        self.gen_loss = losses.GeneratorLoss()
        self.feat_matching_loss = losses.FeatureMatchingLoss()
        self.melspec_loss = losses.MelSpecReconstructionLoss(sample_rate=sample_rate)
        self.mr_melspec_loss = losses.MultiResolutionSTFTLoss(
            fft_sizes, hop_sizes, win_sizes
        )

        if use_sm_loss:
            self.sm_loss = losses.SpeakerSimilarityLoss(
                biometric_model_type, biometric_model_name, sample_rate, loss_device
            )
        else:
            self.sm_loss = None

        if use_wavlm_loss:
            self.wavlm_loss = losses.WavLMLoss(wavlm_model_name, sample_rate)
        else:
            self.wavlm_loss = None

        if use_cdpam_loss:
            self.cdpam_loss = losses.CDPAMLoss(loss_device)
        else:
            self.cdpam_loss = None

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

        if use_clearml_logger:
            self.clearml_task = Task.init(
                task_name=saver.expr_path.name, project_name=saver.expr_path.parent.name
            )
        else:
            self.clearml_task = None

    def on_fit_start(self):
        self.batch_processor.set_device(self.device)
        for loss in [self.sm_loss, self.cdpam_loss]:
            if loss is not None:
                if self.hparams.loss_device == "cpu":
                    loss.set_device(str(self.device))
                else:
                    loss.set_device(self.hparams.loss_device)

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        opt_disc = torch.optim.AdamW(
            disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9)
        )
        opt_gen = torch.optim.AdamW(
            gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9)
        )

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [
                {"scheduler": scheduler_disc, "interval": "step"},
                {"scheduler": scheduler_gen, "interval": "step"},
            ],
        )

    def forward(self, inputs, **kwargs):
        losses = {}

        feats, ft_losses, ft_additional = self.feature_extractor(inputs, **kwargs)
        losses.update(ft_losses)
        kwargs.update(ft_additional)

        x = self.backbone(feats, **kwargs)

        audio_output, mb_audio_output, head_losses = self.head(x, **kwargs)
        losses.update(head_losses)

        return audio_output, mb_audio_output, losses

    @staticmethod
    def _get_kwargs(inputs: VocoderForwardInput):
        return {
            "audio_gt": inputs.waveform.squeeze(-1),
            "ac_latent_gt": inputs.ac_feat,
            "speaker_emb_gt": inputs.speaker_emb,
            "spec_chunk": inputs.additional_inputs.get("spec_chunk"),
            "model_inputs": inputs,
        }

    def log_image(self, tag, snd_tensor):
        img = plot_spectrogram_to_numpy(snd_tensor.cpu().data.numpy())
        self.logger.experiment.add_image(
            tag,
            img,
            self.global_step,
            dataformats="HWC",
        )

        if self.clearml_task is not None:
            self.clearml_task.logger.report_image(
                title=tag,
                image=img,
                iteration=self.global_step,
                series="image color red",
            )

    def log_audio(self, tag, snd_tensor):
        self.logger.experiment.add_audio(
            tag,
            snd_tensor.cpu().float(),
            self.global_step,
            self.hparams.sample_rate,
        )

        if self.clearml_task is not None:
            audio_chunk = AudioChunk(
                data=snd_tensor.cpu().float().data.numpy(), sr=self.hparams.sample_rate
            )
            self.clearml_task.logger.report_media(
                title=tag,
                stream=io.BytesIO(audio_chunk.tobytes()),
                iteration=self.global_step,
                file_extension="wav",
                series="tada",
            )

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )

        audio_input = inputs.waveform.squeeze(-1)
        kwargs = self._get_kwargs(inputs)

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat, _, _ = self(inputs, **kwargs)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(
                y=audio_input,
                y_hat=audio_hat,
            )
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(
                y=audio_input,
                y_hat=audio_hat,
            )
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat, _, inner_losses = self(inputs, **kwargs)
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input,
                    y_hat=audio_hat,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input,
                    y_hat=audio_hat,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(
                    disc_outputs=gen_score_mrd
                )
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(
                    fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp
                ) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(
                    fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd
                ) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            melspec_loss = self.melspec_loss(audio_hat, audio_input)
            mr_melspec_loss = self.mr_melspec_loss(audio_hat, audio_input)
            mel_loss = melspec_loss + mr_melspec_loss

            auxiliary_loss = 0
            if self.global_step % self.hparams.auxiliary_losses_every == 0:
                for loss_fn in [self.sm_loss, self.wavlm_loss, self.cdpam_loss]:
                    if loss_fn is not None:
                        loss_val = loss_fn(audio_hat, audio_input)
                        auxiliary_loss += loss_val.to(self.device)
                        self.log(f"generator/{loss_fn.__class__.__name__}", loss_val)

            for name, value in inner_losses.items():
                self.log(f"generator/{name}", value)
                if value.requires_grad and not name.startswith("constant"):
                    auxiliary_loss += value

            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
                + self.hparams.auxiliary_loss_coeff * auxiliary_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.log_audio(
                    "train/audio_in",
                    audio_input[0],
                )
                self.log_audio(
                    "train/audio_pred",
                    audio_hat[0],
                )

                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))

                self.log_image("train/mel_target", mel)
                self.log_image(
                    "train/mel_pred",
                    mel_hat,
                )

            return loss

    def on_validation_epoch_start(self):
        if self.hparams.evaluate_utmos:
            from tts.vocoders.vocos.metrics.utmos import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        inputs, targets, metadata = self.batch_processor(
            batch, batch_idx, self.global_step
        )

        audio_input = inputs.waveform.squeeze(-1)
        kwargs = self._get_kwargs(inputs)

        audio_hat, _, feat_losses = self(inputs, **kwargs)

        audio_16_khz = torchaudio.functional.resample(
            audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000
        )
        audio_hat_16khz = torchaudio.functional.resample(
            audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000
        )

        if self.hparams.evaluate_periodicty:
            from tts.vocoders.vocos.metrics.periodicity import (
                calculate_periodicity_metrics,
            )

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(
                audio_16_khz, audio_hat_16khz
            )
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(
                audio_16_khz.float().cpu().numpy(), audio_hat_16khz.float().cpu().numpy()
            ):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self.log_audio("val_in", audio_in)
            self.log_audio("val_pred", audio_pred)

            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))

            self.log_image("val_mel_target", mel_target)
            self.log_image(
                "val_mel_hat",
                mel_hat,
            )

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """Override global_step so that it returns the total number of batches
        processed."""
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(
                self.global_step + 1
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self.saver.to_save)

    def on_after_backward(self) -> None:
        # if not self.trainer._detect_anomaly:  # type: ignore
        #    return

        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(
                "Detected inf or NaN values in gradients. not updating model parameters."
            )
            self.zero_grad()
