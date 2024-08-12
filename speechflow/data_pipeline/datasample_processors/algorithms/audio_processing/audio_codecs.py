import enum
import typing as tp
import logging

from pathlib import Path

import dac
import numpy as np
import torch

from speechflow.data_pipeline.datasample_processors.data_types import AudioCodecFeatures
from speechflow.io import AudioChunk
from speechflow.utils.fs import get_root_dir
from speechflow.utils.profiler import Profiler

__all__ = [
    "DAC",
    "VocosAC",
]

LOGGER = logging.getLogger("root")


class ACFeatureType(enum.Enum):
    """
    "latent" : Tensor[B x N*D x T]
        Projected latents (continuous representation of input before quantization)
    "quantized" : Tensor[B x N x T]
        Codebook indices for each codebook (quantized discrete representation of input)
    "continuous" : Tensor[B x D x T]
        Quantized continuous representation of input
    """

    latent = 0
    quantized = 1
    continuous = 2


class BaseAudioCodecModel(torch.nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        feat_type: ACFeatureType = ACFeatureType.continuous,
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
    ):
        super().__init__()

        self.device = device
        self.sample_rate = 24000
        self.embedding_dim = 0

        self._feat_type = (
            ACFeatureType[feat_type] if isinstance(feat_type, str) else feat_type
        )

        self._min_audio_duration = min_audio_duration
        self._max_audio_duration = max_audio_duration

        if self._max_audio_duration is not None:
            self._max_audio_duration = int(max_audio_duration * self.sample_rate)

    def preprocess(self, audio_chunk: AudioChunk) -> torch.Tensor:
        assert np.issubdtype(
            audio_chunk.dtype, np.floating
        ), "Audio data must be floating-point!"

        audio_chunk = audio_chunk.resample(sr=self.sample_rate, fast=True)

        if self._max_audio_duration is not None:
            data = torch.tensor(
                audio_chunk.waveform[: self._max_audio_duration], device=self.device
            )
        else:
            data = torch.tensor(audio_chunk.waveform, device=self.device)

        return data.unsqueeze(0)

    def postprocessing(self, feat: AudioCodecFeatures) -> AudioCodecFeatures:
        if self._min_audio_duration is not None:
            pass
        return feat


class DAC(BaseAudioCodecModel):
    def __init__(
        self,
        device: str = "cpu",
        feat_type: ACFeatureType = ACFeatureType.continuous,
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
    ):
        super().__init__(device, feat_type, min_audio_duration, max_audio_duration)

        self.sample_rate = 24000

        if self._feat_type == ACFeatureType.latent:
            self.embedding_dim = 256
        elif self._feat_type == ACFeatureType.quantized:
            self.embedding_dim = 32
        elif self._feat_type == ACFeatureType.continuous:
            self.embedding_dim = 1024
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        model_path = dac.utils.download(model_type=f"{self.sample_rate // 1000}khz")
        self.model = dac.DAC.load(model_path.as_posix())
        self.model.to(device)

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk, **kwargs) -> AudioCodecFeatures:
        ac_feat = AudioCodecFeatures()
        data = self.preprocess(audio_chunk)

        data = data.unsqueeze(1)

        x = self.model.preprocess(data, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)

        if self._feat_type == ACFeatureType.latent:
            ac_feat.encoder_feat = latents.squeeze(0).t().cpu()
        elif self._feat_type == ACFeatureType.quantized:
            ac_feat.encoder_feat = codes.squeeze(0).t().cpu()
        elif self._feat_type == ACFeatureType.continuous:
            ac_feat.encoder_feat = z.squeeze(0).t().cpu()
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        return self.postprocessing(ac_feat)

    @torch.no_grad()
    def encode(self, data: torch.Tensor):
        x = self.model.preprocess(data, self.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        return self.model.decode(z)


class VocosAC(BaseAudioCodecModel):
    def __init__(
        self,
        ckpt_path: Path,
        device: str = "cpu",
        feat_type: ACFeatureType = ACFeatureType.continuous,
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
    ):
        from tts.vocoders.eval_interface import VocoderEvaluationInterface

        super().__init__(device, feat_type, min_audio_duration, max_audio_duration)

        self.sample_rate = 24000

        self.voc = VocoderEvaluationInterface(ckpt_path=ckpt_path, device=device)
        dim = self.voc.model.feature_extractor.vq_enc.params.encoder_output_dim

        if self._feat_type == ACFeatureType.latent:
            self.embedding_dim = dim
        elif self._feat_type == ACFeatureType.quantized:
            self.embedding_dim = (
                self.voc.model.feature_extractor.vq_codes.params.vq_num_quantizers
            )
        elif self._feat_type == ACFeatureType.continuous:
            self.embedding_dim = dim
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk, **kwargs) -> AudioCodecFeatures:
        from tts.vocoders.data_types import VocoderForwardInput

        ac_feat = AudioCodecFeatures()
        vq_only = kwargs.get("vq_only", False)

        ds = kwargs.get("ds").copy()
        ds.to_tensor()

        lens = torch.LongTensor([ds.magnitude.shape[0]]).to(self.device)

        _input = VocoderForwardInput(
            spectrogram=ds.mel.unsqueeze(0).to(self.device),
            spectrogram_lengths=lens,
            linear_spectrogram=ds.magnitude.unsqueeze(0).to(self.device),
            linear_spectrogram_lengths=lens,
            ssl_feat=ds.ssl_feat.encoder_feat.unsqueeze(0).to(self.device),
            ssl_feat_lengths=lens,
            energy=ds.energy.unsqueeze(0).to(self.device),
            pitch=ds.pitch.unsqueeze(0).to(self.device),
            speaker_emb=ds.speaker_emb.to(self.device),
            lang_id=torch.LongTensor([self.voc.lang_id_map[ds.lang]] * len(lens)).to(
                self.device
            ),
            additional_inputs={
                "style_embedding": ds.additional_fields["style_embedding"]
                .unsqueeze(0)
                .to(self.device)
            },
        )
        _output = self.voc.model.inference(_input, vq_only=vq_only)
        _addc = _output.additional_content

        if self._feat_type == ACFeatureType.latent:
            ac_feat.encoder_feat = _addc["vq_latent"].queeze(0).cpu()
        elif self._feat_type == ACFeatureType.quantized:
            ac_feat.encoder_feat = _addc["vq_codes"].squeeze(0).cpu()
        elif self._feat_type == ACFeatureType.continuous:
            ac_feat.encoder_feat = _addc["vq_z"].squeeze(0).cpu()
        else:
            raise NotImplementedError(f"feature {self._feat_type} is not supported")

        if not vq_only:
            ac_feat.waveform = _output.waveform[0].squeeze(0).cpu()

        return self.postprocessing(ac_feat)


if __name__ == "__main__":
    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(_wav_path, end=3.9).load()

    for _feat_type in [
        ACFeatureType.continuous,
        ACFeatureType.quantized,
        ACFeatureType.latent,
    ]:
        for _ac_cls in [DAC]:
            try:
                _ac_model = _ac_cls(feat_type=_feat_type)
            except Exception as e:
                print(e)
                continue

            with Profiler(_ac_cls.__name__) as prof:
                _ac_feat = _ac_model(_audio_chunk)

            print(f"{_ac_cls.__name__}: {_ac_feat.encoder_feat.shape}")
            assert _ac_feat.encoder_feat.shape[-1] == _ac_model.embedding_dim
