import typing as tp

import torch

from torch.nn import functional as F

from tts.acoustic_models.data_types import TTSForwardOutput
from tts.acoustic_models.models.tts_model import ParallelTTSModel
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor

__all__ = ["TTSFeatures"]


class TTSFeatures(FeatureExtractor):
    def __init__(
        self,
        tts_cfg: tp.MutableMapping,
        output_dim: int = 256,
    ):
        super().__init__()
        self._tts = ParallelTTSModel(tts_cfg)
        self._mel_proj = torch.nn.Linear(output_dim, 80)

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        if inputs.__class__.__name__ != "TTSForwardInputWithSSML":
            outputs: TTSForwardOutput = self._tts(inputs)
            losses = outputs.additional_losses
            additional_content = outputs.additional_content
        else:
            outputs = inputs  # type: ignore
            losses = {}
            additional_content = {}

        if self.training:
            target_spec = inputs.spectrogram
            if isinstance(outputs.spectrogram, list):
                for idx, predict_spec in enumerate(outputs.spectrogram):
                    if predict_spec.shape[-1] == target_spec.shape[-1]:
                        losses[f"spec_loss_{idx}"] = F.l1_loss(predict_spec, target_spec)
                    else:
                        losses[f"spec_loss_{idx}"] = F.l1_loss(
                            self._mel_proj(predict_spec), target_spec
                        )

                x = outputs.spectrogram[-1]
            else:
                losses["spec_loss"] = F.l1_loss(outputs.spectrogram, target_spec)
                x = outputs.spectrogram
        else:
            if isinstance(outputs.spectrogram, list):
                x = outputs.spectrogram[-1]
            else:
                x = outputs.spectrogram

        if "spec_chunk" in inputs.additional_inputs:
            chunk = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                chunk.append(x[i, a:b, :])

            output = torch.stack(chunk)
        else:
            output = x

        return output.transpose(1, -1), losses, additional_content
