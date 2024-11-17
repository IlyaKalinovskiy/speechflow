import numpy as np
import torch
import numpy.typing as npt
import pytorch_lightning as pl

from pytorch_lightning import Callback

from speechflow.utils.plotting import figure_to_ndarray, plot_1d

__all__ = ["GradNormCallback", "VisualizerCallback"]


class GradNormCallback(Callback):
    """Callback to log the gradient norm."""

    @staticmethod
    def _gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
        """Compute the gradient norm.

        Args:
            model (Module): PyTorch modules.
            norm_type (float, optional): Type of the norm. Defaults to 2.0.

        Returns:
            Tensor: Gradient norm.

        """
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type
        )
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", self._gradient_norm(model))


class VisualizerCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx != 0:
            return

        with torch.no_grad():
            inputs, targets, metadata = pl_module.batch_processor(batch, batch_idx, 0)
            _, _, ft_additional = pl_module.feature_extractor(inputs)

        batch_size = targets.spectrogram.size(0)

        if batch_size <= 1:
            random_idx = 0
        else:
            random_idx = np.random.randint(0, batch_size - 1)

        for name in ["energy", "pitch"]:
            target = getattr(targets, name)[random_idx]
            predict = ft_additional[f"{name}_predict"][random_idx]
            data = torch.stack([target, predict]).cpu().numpy()
            self._log_1d_signal(pl_module, data, f"{name}_predict")

    @staticmethod
    def _log_1d_signal(
        module: pl.LightningModule,
        signal: npt.NDArray,
        tag: str,
    ):
        fig_to_plot = plot_1d(signal)
        data_to_log = figure_to_ndarray(fig_to_plot)
        module.log_image(tag, None, fig=data_to_log.swapaxes(0, 2))
