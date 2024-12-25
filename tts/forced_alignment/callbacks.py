import random
import typing as tp

import numpy as np
import torch
import numpy.typing as npt
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback

from speechflow.utils.plotting import (
    figure_to_ndarray,
    phonemes_to_frame_ticks,
    plot_spectrogram,
)


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def _plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(12, 4))  # type: ignore
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class AligningVisualisationCallback(Callback):
    def __init__(self):
        pass

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx == 0:
            inputs, outputs, _, _ = pl_module.test_step(batch, batch_idx)
            aligning = outputs.aligning_path

            batch_size = aligning.size(0)
            random_idx = random.randint(0, batch_size - 1)

            spec_lens = inputs.output_lengths.detach().cpu().numpy().tolist()
            phonemes = list(
                batch.collated_samples.additional_fields["transcription_text"][random_idx]
            )
            spectrogram = (
                inputs.spectrogram[random_idx, : spec_lens[random_idx]]
                .T.detach()
                .cpu()
                .numpy()
            )
            alignment = (
                outputs.aligning_path[
                    random_idx, : spec_lens[random_idx], : len(phonemes)
                ]
                .detach()
                .cpu()
                .numpy()
            )

            scale = float(spec_lens[random_idx] / outputs.output_lengths[random_idx])
            self._log_target(
                pl_module, trainer, spectrogram, alignment, phonemes, scale=scale
            )
            self._log_aligning(pl_module, trainer, np.transpose(alignment))

    @staticmethod
    def _log_target(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        spectrogram: npt.NDArray,
        alignment: npt.NDArray,
        phonemes: tp.List[str],
        name: str = "TargetSpectrogramWithAlignment",
        scale: float = 1.0,
    ):
        frame_ticks = phonemes_to_frame_ticks(alignment, phonemes)
        frame_ticks = [t * scale for t in frame_ticks]
        fig_to_plot = plot_spectrogram(spectrogram, phonemes, frame_ticks)
        data_to_log = figure_to_ndarray(fig_to_plot)

        module.logger.experiment.add_image(
            name,
            data_to_log,
            trainer.global_step,
            dataformats="CHW",
        )

    @staticmethod
    def _log_aligning(
        module: pl.LightningModule,
        trainer: pl.Trainer,
        aligning: torch.Tensor,
        name: str = "Aligning",
    ):
        module.logger.experiment.add_image(
            name,
            _plot_alignment_to_numpy(aligning),
            trainer.global_step,
            dataformats="HWC",
        )
