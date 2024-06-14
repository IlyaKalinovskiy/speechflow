import json
import math
import typing as tp
import logging

from multiprocessing import current_process
from pathlib import Path

import numpy as np
import torch
import whisper
import torchaudio
import transformers

from transformers.tokenization_utils_base import PaddingStrategy
from whisper.decoding import DecodingTask

from speechflow.data_pipeline.datasample_processors.data_types import SSLFeatures
from speechflow.io import AudioChunk, check_path, tp_PATH
from speechflow.utils.fs import get_root_dir
from speechflow.utils.profiler import Profiler

LOGGER = logging.getLogger("root")

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError as e:
    if current_process().name == "MainProcess":
        LOGGER.warning(f"speechbrain is not available: {e}")


__all__ = [
    "Whisper",
    "Wav2Vec",
    "Hubert",
    "WavLM",
    "ECAPABiometric",
]


class BaseSSLModel(torch.nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
    ):
        super().__init__()

        self.device = device
        self.sample_rate = 16000
        self.embedding_dim = 0

        self._min_audio_duration = min_audio_duration
        self._max_audio_duration = max_audio_duration

        if self._max_audio_duration is not None:
            self._max_audio_duration = int(max_audio_duration * self.sample_rate)

    def preprocessing(self, audio_chunk: AudioChunk) -> torch.Tensor:
        assert np.issubdtype(
            audio_chunk.dtype, np.floating
        ), "Audio data must be floating-point!"

        # if audio_chunk.sr != self.sample_rate:
        #     LOGGER.warning(
        #         trace(
        #             self,
        #             message=f"Only {self.sample_rate} sample rate is available "
        #                     f"but got sample rate={audio_chunk.sr}!",
        #             full=False,
        #         ),
        #     )

        audio_chunk = audio_chunk.resample(sr=self.sample_rate, fast=True)

        if self._min_audio_duration is not None:
            pass

        if self._max_audio_duration is not None:
            data = torch.tensor(
                audio_chunk.waveform[: self._max_audio_duration], device=self.device
            )
        else:
            data = torch.tensor(audio_chunk.waveform, device=self.device)

        return data.unsqueeze(0)

    def postprocessing(self, feat: SSLFeatures) -> SSLFeatures:
        if self._min_audio_duration is not None:
            pass
        return feat


class Whisper(BaseSSLModel):
    def __init__(
        self,
        device: str = "cpu",
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
        model_name: str = "tiny",  # tiny, base, small, medium, large
    ):
        super().__init__(device, min_audio_duration, max_audio_duration)

        self.sample_rate = 16000
        self.embedding_dim = 384
        self.model = whisper.load_model(model_name, device)
        self.options = whisper.DecodingOptions(fp16=False)
        self.dec_task = DecodingTask(self.model, self.options)
        self.pos_emb = self.model.encoder.positional_embedding.clone()

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        mel = whisper.log_mel_spectrogram(data.squeeze(0)).unsqueeze(0)

        assert mel.shape[-1] <= self.pos_emb.shape[0]
        self.model.encoder.positional_embedding = self.pos_emb[
            : math.ceil(mel.shape[-1] / 2)
        ]
        emb = self.dec_task._get_audio_features(mel)
        ssl_feat.encode = emb.squeeze(0).cpu()

        return self.postprocessing(ssl_feat)


class Wav2Vec(BaseSSLModel):
    def __init__(
        self,
        device: str = "cpu",
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
        model_name: tp.Optional[
            tp.Union[str, Path]
        ] = "anton-l/wav2vec2-large-xlsr-53-russian",
        pretrain_path: tp.Optional[tp_PATH] = None,
        feature_type: str = "encoder",
        level: int = 4,
        stream_mod: tp.Optional[dict] = None,
        trim_pad: bool = True,
    ):
        super().__init__(device, min_audio_duration, max_audio_duration)

        self.sample_rate = 16000
        self.embedding_dim = 1024
        self._feature_type = feature_type
        self._level = level
        self._stream_mod = stream_mod
        self._trim_pad = trim_pad

        self._init_model(model_name, feature_type, pretrain_path)

    def _init_model(
        self, model_name: tp.Union[str, Path], feature_type: str, pretrain_path: tp_PATH
    ):
        self.model = transformers.Wav2Vec2Model.from_pretrained(
            model_name,
            attention_dropout=0.01,
            hidden_dropout=0.01,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.01,
        )
        self.processor = transformers.Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sample_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        if pretrain_path is not None:
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            state_dict = {
                k.replace("model.model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if "lm_head" not in k and "phoneme_proj" not in k
            }
            self.model.load_state_dict(state_dict, strict=True)

        if feature_type in ["encoder", "encode_level"]:
            self.model.eval()
            self.model.to(self.device)
        elif feature_type == "projection":
            self.wav_ft_model = self.model.feature_extractor
            self.wav_fp_model = self.model.feature_projection

            self.wav_ft_model.eval()
            self.wav_fp_model.eval()

            self.wav_ft_model.to(self.device)
            self.wav_fp_model.to(self.device)
        else:
            raise ValueError(
                f"Available model_type's: wav2vec encoder, wav2vec projection, "
                f"but got model_type={feature_type}!"
            )

    def encode(
        self,
        input_values: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        extract_features = self.model.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.model.feature_projection(extract_features)

        position_embeddings = self.model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.encoder.dropout(hidden_states)

        for layer in self.model.encoder.layers[:level]:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]

        return hidden_states

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        processed = self.processor(
            data.squeeze(0),
            padding=PaddingStrategy.DO_NOT_PAD,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        processed = {k: v.to(self.device) for k, v in processed.data.items()}

        tmp_attention_mask = None
        if self._stream_mod is not None:
            wave = processed["input_values"].squeeze(0)
            pad_size = wave.shape[0] % self._stream_mod["chunk_size"]
            a = torch.zeros(self._stream_mod["context_size"][0]).to(self.device)
            b = torch.zeros(self._stream_mod["context_size"][1]).to(self.device)
            c = torch.zeros(pad_size).to(self.device)
            wave_pad = torch.cat([a, wave, c, b])
            chunks_size = self._stream_mod["chunk_size"] + sum(
                self._stream_mod["context_size"]
            )
            chunks = wave_pad.unfold(0, chunks_size, self._stream_mod["chunk_size"])
            processed["input_values"] = chunks
            tmp_attention_mask = processed.pop("attention_mask", None)

        if self._feature_type == "encoder":
            outputs = self.model(**processed)
            ssl_feat.encode = outputs[0]
            ssl_feat.attention_mask = processed.get("attention_mask")
        elif self._feature_type == "projection":
            # Wav2Vec2FeatureExtractor
            extract_features = self.wav_ft_model(processed["input_values"])

            extract_features = extract_features.transpose(1, 2)
            if processed["attention_mask"] is not None:
                # compute reduced attention_mask corresponding to feature vectors
                attention_mask = self.model._get_feature_vector_attention_mask(
                    extract_features.shape[1],
                    processed["attention_mask"],
                    add_adapter=False,
                )
                # print(attention_mask.shape, extract_features.shape)
            else:
                attention_mask = None

            # Wav2Vec2FeatureProjection
            hidden_states, extract_features = self.wav_fp_model(extract_features)
            hidden_states = self.model._mask_hidden_states(
                hidden_states,
                mask_time_indices=None,
                attention_mask=attention_mask,
            )
            ssl_feat.encode = hidden_states
            ssl_feat.attention_mask = attention_mask
        elif self._feature_type == "encode_level":
            if self._level > 0:
                ssl_feat.encode = self.encode(
                    input_values=processed["input_values"],
                    level=self._level,
                )
            else:
                outputs = self.model(
                    input_values=processed["input_values"],
                    attention_mask=processed["attention_mask"],
                    output_hidden_states=True,
                )
                ssl_feat.encode = outputs.hidden_states[self._level]
            ssl_feat.attention_mask = processed["attention_mask"]
        else:
            raise ValueError(
                f"Available model_type's: wav2vec encoder, wav2vec projection, encode_level "
                f"but got model_type={self._feature_type}!"
            )

        if self._stream_mod is not None:
            chunks_size = self._stream_mod["chunk_size"] + sum(
                self._stream_mod["context_size"]
            )
            shape = ssl_feat.encode.shape
            a = round(shape[1] * self._stream_mod["context_size"][0] / chunks_size)
            b = -round(shape[1] * self._stream_mod["context_size"][1] / chunks_size)
            ssl_feat.encode = ssl_feat.encode[:, a:b, :].reshape(
                (1, -1, ssl_feat.encode.shape[2])
            )
            feat_len = round(audio_chunk.duration / 0.02)
            ssl_feat.encode = ssl_feat.encode[:, :feat_len, :]
            ssl_feat.attention_mask = tmp_attention_mask

        ssl_feat.encode = ssl_feat.encode.cpu().squeeze(0)
        ssl_feat.attention_mask = ssl_feat.attention_mask.cpu().bool().squeeze(0)

        if self._trim_pad:
            ratio = ssl_feat.attention_mask.shape[0] / ssl_feat.encode.shape[0]
            attention_len = int(ssl_feat.attention_mask.sum())
            feat_len = int(attention_len / ratio)
            ssl_feat.encode = ssl_feat.encode[:feat_len, :].contiguous()
            ssl_feat.attention_mask = ssl_feat.attention_mask[:attention_len].contiguous()

        ssl_feat.encode = ssl_feat.encode.cpu()
        return self.postprocessing(ssl_feat)


class Hubert(Wav2Vec):
    def _init_model(self, model_name, feature_type, pretrain_path):
        self.model = transformers.HubertForCTC.from_pretrained(model_name)
        self.model.lm_head = torch.nn.Linear(1024, 64)
        self.processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        if pretrain_path is not None:
            checkpoint = torch.load(pretrain_path, map_location="cpu")
            state_dict = {
                k.replace("model.wav2vec2.", "hubert.").replace("aux", "lm_head"): v
                for k, v in checkpoint["state_dict"].items()
                if "logit_generator" not in k
            }
            state_dict = {
                k.replace("encoder.feature_projection", "feature_projection").replace(
                    "transformer.", ""
                ): v
                for k, v in state_dict.items()
            }
            state_dict = {
                k.replace(
                    "model.mask_generator.mask_embedding", "hubert.masked_spec_embed"
                ): v
                for k, v in state_dict.items()
            }
            state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
            # bias mismatch for feature extraction is correct
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(e)
                self.model.load_state_dict(state_dict, strict=False)

        if feature_type in ["encoder", "encode_level"]:
            self.model.eval()
            self.model.to(self.device)
        else:
            raise ValueError(
                f"Available model_type's: hubert encoder, "
                f"but got model_type={feature_type}!"
            )


class WavLM(BaseSSLModel):
    def __init__(
        self,
        device: str = "cpu",
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
        model_name: tp.Literal["WAVLM_BASE_PLUS", "WAVLM_LARGE"] = "WAVLM_LARGE",
        model_dir: tp.Optional[tp_PATH] = None,
        num_layer: int = 9,
    ):
        super().__init__(device, min_audio_duration, max_audio_duration)
        """
        num_layer: 9 - asr task, base+; -1 asr task large
        more details: https://arxiv.org/pdf/2110.13900.pdf (see Fig. 2)
        """
        self._num_layer = num_layer

        pipe = getattr(torchaudio.pipelines, model_name)

        if model_dir is not None and model_dir.exists():
            self._model = pipe.get_model(dl_kwargs={"model_dir": model_dir}).to(
                self.device
            )
        else:
            self._model = pipe.get_model().to(self.device)

        self.sample_rate = pipe.sample_rate
        self.embedding_dim = pipe._params["encoder_embed_dim"]

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        feat = self._model.extract_features(data)[0]

        if self._num_layer:
            emb = feat[self._num_layer]
        else:
            emb = feat

        ssl_feat.encode = emb.squeeze(0).cpu()

        return self.postprocessing(ssl_feat)


class ECAPABiometric(BaseSSLModel):
    def __init__(
        self,
        device: str = "cpu",
        min_audio_duration: tp.Optional[float] = None,
        max_audio_duration: tp.Optional[float] = None,
        model_name: tp.Optional[tp.Union[str, Path]] = "spkrec-ecapa-voxceleb",
    ):
        super().__init__(device, min_audio_duration, max_audio_duration)

        self.sample_rate = 16000
        self.embedding_dim = 192

        if not Path(model_name).exists():
            model_name = (
                get_root_dir()
                / f"speechflow/data/temp/biometric/speechbrain/{model_name}"
            )
            model_name.mkdir(parents=True, exist_ok=True)

        self._model = EncoderClassifier.from_hparams(
            source=f"speechbrain/{Path(model_name).name}",
            savedir=Path(model_name).absolute().as_posix(),
            run_opts={"device": self.device},
        )

        _model = self._model.mods.embedding_model
        _state_dict = _model.fc.state_dict()
        _state_dict_mean = {
            "weight": _state_dict["conv.weight"][:, :3072, :],
            "bias": _state_dict["conv.bias"],
        }
        _state_dict_bn = {
            "weight": _model.asp_bn.state_dict()["norm.weight"][:3072],
            "bias": _model.asp_bn.state_dict()["norm.bias"][:3072],
            "running_mean": _model.asp_bn.state_dict()["norm.running_mean"][:3072],
            "running_var": _model.asp_bn.state_dict()["norm.running_var"][:3072],
        }

        self.proj = torch.nn.Conv1d(3072, 192, 1, device=device)
        self.bn = torch.nn.BatchNorm1d(3072, device=device)
        self.proj.load_state_dict(_state_dict_mean)
        self.bn.load_state_dict(_state_dict_bn)

    @torch.inference_mode()
    def get_feats(self, wavs, wav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        feats = self._model.mods.compute_features(wavs)
        feats = self._model.mods.mean_var_norm(feats, wav_lens)

        return feats

    @torch.inference_mode()
    def get_embeddings(self, features):
        _model = self._model.mods.embedding_model
        x = features.transpose(1, 2)

        xl = []
        for layer in _model.blocks:
            try:
                x = layer(x, lengths=None)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = _model.mfa(x)
        x = self.bn(x)
        x = self.proj(x).transpose(1, 2)

        return x

    @torch.inference_mode()
    def __call__(self, audio_chunk: AudioChunk) -> SSLFeatures:
        ssl_feat = SSLFeatures()
        data = self.preprocessing(audio_chunk)

        features = self.get_feats(data).to(self.device)
        emb = self.get_embeddings(features=features)
        ssl_feat.encode = emb.squeeze(0).cpu()

        return self.postprocessing(ssl_feat)


if __name__ == "__main__":
    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _audio_chunk = AudioChunk(_wav_path, end=3.9).load()

    for _ssl_cls in [Whisper, Wav2Vec, WavLM, ECAPABiometric]:
        try:
            _ssl_model = _ssl_cls()
        except Exception as e:
            print(e)
            continue

        with Profiler(_ssl_cls.__name__) as prof:
            _ssl_feat = _ssl_model(_audio_chunk)

        print(f"{_ssl_cls.__name__}: {_ssl_feat.encode.shape}")
        assert _ssl_feat.encode.shape[-1] == _ssl_model.embedding_dim
