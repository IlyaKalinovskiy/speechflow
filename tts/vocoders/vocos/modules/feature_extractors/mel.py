import torch
import torchaudio

from tts.vocoders.data_types import VocoderForwardInput
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor
from tts.vocoders.vocos.utils.tensor_utils import safe_log

__all__ = ["MelSpectrogramFeatures"]


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self, sample_rate=24000, n_fft=1024, hop_length=320, n_mels=80, padding="center"
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, inputs: VocoderForwardInput, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(
                inputs.waveform, (pad // 2, pad // 2), mode="reflect"
            )
        else:
            audio = inputs.waveform

        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features, {}
