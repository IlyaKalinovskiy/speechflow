import typing as tp

import torch

from torch.nn import functional as F

from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
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
            energy_target = outputs.additional_content["energy_postprocessed"]
            pitch_target = outputs.additional_content["pitch_postprocessed"]

            chunk = []
            energy = []
            pitch = []
            for i, (a, b) in enumerate(inputs.additional_inputs["spec_chunk"]):
                chunk.append(x[i, a:b, :])
                energy.append(energy_target[i, a:b])
                pitch.append(pitch_target[i, a:b])

            output = torch.stack(chunk)
            additional_content["energy"] = torch.stack(energy)
            additional_content["pitch"] = torch.stack(pitch)
            additional_content["style_emb"] = outputs.additional_content[
                "style_emb"
            ].squeeze(1)
        else:
            output = x
            if isinstance(outputs, TTSForwardOutput):
                d = outputs.additional_content
            elif isinstance(outputs, TTSForwardInput):
                d = outputs.additional_inputs

            additional_content["energy"] = d["energy_postprocessed"]
            additional_content["pitch"] = d["pitch_postprocessed"]
            additional_content["style_emb"] = d["style_emb"].squeeze(1)

        return output.transpose(1, -1), losses, additional_content
