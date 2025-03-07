import typing as tp

import torch

from torch.nn import functional as F

from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.acoustic_models.models.tts_model import ParallelTTSModel, ParallelTTSParams
from tts.acoustic_models.modules.common.blocks import Regression
from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor

__all__ = ["TTSFeatures", "TTSFeaturesParams"]


class TTSFeaturesParams(ParallelTTSParams):
    pass


class TTSFeatures(FeatureExtractor):
    def __init__(self, params: tp.Union[tp.MutableMapping, TTSFeaturesParams]):
        super().__init__(params)
        self.tts_model = ParallelTTSModel(params)
        if params.decoder_output_dim != params.mel_spectrogram_dim:
            self.proj = Regression(params.decoder_output_dim, params.mel_spectrogram_dim)
        else:
            self.proj = torch.nn.Identity()

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        if inputs.__class__.__name__ != "TTSForwardInputWithSSML":
            outputs: TTSForwardOutput = self.tts_model(inputs)
            losses = outputs.additional_losses
            additional_content = outputs.additional_content
        else:
            outputs = inputs  # type: ignore
            losses = {}
            additional_content = {}

        if isinstance(outputs.spectrogram, list) or outputs.spectrogram.ndim == 4:
            x = outputs.spectrogram[0]
        else:
            x = outputs.spectrogram

        if self.training:
            target_spec = inputs.spectrogram
            losses["spectral_loss"] = F.l1_loss(self.proj(x), target_spec)

        if "spec_chunk" in inputs.additional_inputs:
            if kwargs.get("discriminator_step", False):
                energy_target = inputs.pitch
                pitch_target = inputs.energy
            else:
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
            additional_content["energy"] = torch.stack(energy).squeeze(-1)
            additional_content["pitch"] = torch.stack(pitch).squeeze(-1)
            additional_content["condition_emb"] = outputs.additional_content[
                "style_emb"
            ].squeeze(1)
        else:
            output = x
            if isinstance(outputs, TTSForwardOutput):
                d = outputs.additional_content
            elif isinstance(outputs, TTSForwardInput):
                d = outputs.additional_inputs
            else:
                raise TypeError(f"Type {type(outputs)} is not supported.")

            additional_content["energy"] = d["energy_postprocessed"].squeeze(-1)
            additional_content["pitch"] = d["pitch_postprocessed"].squeeze(-1)
            additional_content["condition_emb"] = d["style_emb"][:, 0, :]

        return output.transpose(1, -1), losses, additional_content
