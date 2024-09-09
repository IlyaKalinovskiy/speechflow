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

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        outputs: TTSForwardOutput = self._tts(inputs)

        target_spec = inputs.spectrogram
        losses = outputs.additional_losses

        if isinstance(outputs.spectrogram, list):
            for idx, predict_spec in enumerate(outputs.spectrogram):
                if predict_spec.shape[-1] == target_spec.shape[-1]:
                    losses[f"spec_loss_{idx}"] = F.l1_loss(predict_spec, target_spec)

            x = outputs.spectrogram[-1]
        else:
            losses["spec_loss"] = F.l1_loss(outputs.spectrogram, target_spec)
            x = outputs.spectrogram

        if "spec_chunk" in inputs.additional_inputs:
            chunk = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                chunk.append(x[i, a:b, :])

            output = torch.stack(chunk)
        else:
            output = x

        return output.transpose(1, -1), losses, outputs.additional_content
