import typing as tp

from copy import deepcopy
from dataclasses import dataclass
from os import environ as env

import numpy as np
import torch
import numpy.typing as npt

from speechflow.data_pipeline.collate_functions.spectrogram_collate import (
    SpectrogramCollateOutput,
)
from speechflow.data_pipeline.core import PipelineComponents
from speechflow.data_pipeline.datasample_processors import SignalProcessor
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    TTSDataSample,
)
from speechflow.io import AudioChunk, Config, check_path, tp_PATH
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from tts.acoustic_models.data_types import TTSForwardInput, TTSForwardOutput
from tts.vocoders.data_types import VocoderForwardInput, VocoderForwardOutput
from tts.vocoders.vocos.pretrained import Vocos

__all__ = ["VocoderEvaluationInterface", "VocoderOptions"]


@dataclass
class VocoderOptions:
    pass

    def copy(self) -> "VocoderOptions":
        return deepcopy(self)


class VocoderLoader:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        ckpt_path: tp_PATH,
        device: str = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
        **kwargs,
    ):
        env["DEVICE"] = device

        self.ckpt_path = ckpt_path

        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(self.ckpt_path)
        else:
            checkpoint = ckpt_preload

        self.cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(
            checkpoint
        )

        self.pipe = self._load_data_pipeline(self.cfg_data)
        self.pipe_for_reference = self.pipe.with_ignored_handlers(
            ignored_data_handlers={"SSLProcessor", "MultilingualPLBert"}
        )
        self.lang_id_map = checkpoint.get("lang_id_map", {})
        self.speaker_id_map = checkpoint.get("speaker_id_map", {})

        cfg_model["model"]["feature_extractor"]["init_args"].n_langs = len(
            self.lang_id_map
        )
        cfg_model["model"]["feature_extractor"]["init_args"].n_speakers = len(
            self.speaker_id_map
        )

        # Load model
        if self._check_vocos_signature(checkpoint):
            self.model = self._load_vocos_model(cfg_model, checkpoint)
        else:
            raise NotImplementedError

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

    @staticmethod
    def _check_vocos_signature(checkpoint: tp.Dict) -> bool:
        return "Vocos" in checkpoint["files"]["model.yml"]

    @staticmethod
    def _load_data_pipeline(cfg_data: Config) -> PipelineComponents:
        cfg_data["processor"].pop("dump", None)
        cfg_data["singleton_handlers"]["handlers"] = []
        pipe = PipelineComponents(Config(cfg_data).trim("ml"), data_subset_name="test")
        return pipe.with_ignored_fields(
            ignored_data_fields={"sent", "phoneme_timestamps"}
        ).with_ignored_handlers(
            ignored_data_handlers={"SignalProcessor", "WaveAugProcessor", "normalize"}
        )

    @staticmethod
    def _load_vocos_model(cfg_model: Config, checkpoint: tp.Dict[str, tp.Any]) -> Vocos:
        model = Vocos.init_from_config(cfg_model["model"])
        model.eval()

        state_dict: tp.Dict[str, tp.Any] = checkpoint["state_dict"]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "discriminators" not in k and "loss" not in k
        }
        try:
            model.load_state_dict(state_dict)
        except Exception:
            state_dict = {k.replace("lstms.", "rnns."): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

        model.head.remove_weight_norm()
        return model


class VocoderEvaluationInterface(VocoderLoader):
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        ckpt_path: tp_PATH,
        device: str = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(ckpt_path, device, ckpt_preload, **kwargs)
        self.sample_rate = find_field(self.cfg_data, "sample_rate")
        self.hop_len = find_field(self.cfg_data, "hop_len")
        self.n_mels = find_field(self.cfg_data, "n_mels")
        self.preemphasis_coef = self.find_preemphasis_coef(self.cfg_data)

    @staticmethod
    def find_preemphasis_coef(cfg_data: Config):
        beta = None
        for item in cfg_data["preproc"]["pipe_cfg"].values():
            if isinstance(item, dict) and item.get("type", None) == "SignalProcessor":
                if "preemphasis" in item.get("pipe", []):
                    if item["pipe_cfg"].get("preemphasis"):
                        beta = item["pipe_cfg"]["preemphasis"].get("beta", 0.97)
                    else:
                        beta = 0.97
        return beta

    @torch.inference_mode()
    def evaluate(
        self, inputs: VocoderForwardInput, opt: VocoderOptions
    ) -> VocoderForwardOutput:
        inputs.to(self.device)
        outputs = self.model.inference(inputs)

        waveforms = []
        for idx in range(outputs.waveform.shape[0]):
            waveforms.append(outputs.waveform[idx].cpu().numpy())

        waveforms = np.concatenate(waveforms)

        if self.preemphasis_coef is not None:
            waveforms = self.inv_preemphasis(waveforms, self.preemphasis_coef)

        outputs.audio_chunk = AudioChunk(data=waveforms, sr=self.sample_rate)
        return outputs

    @staticmethod
    def inv_preemphasis(waveform: npt.NDArray, beta: float = 0.97) -> npt.NDArray:
        audio_chunk = AudioChunk(data=waveform, sr=1)
        inv_wave = SignalProcessor.inv_preemphasis(
            AudioDataSample(audio_chunk=audio_chunk), beta=beta
        ).audio_chunk.waveform
        return inv_wave

    def synthesize(
        self,
        tts_input: TTSForwardInput,
        tts_output: TTSForwardOutput,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: VocoderOptions = VocoderOptions(),
    ) -> VocoderForwardOutput:
        voc_in = VocoderForwardInput.init_from_tts(tts_input, tts_output)
        voc_in.lang_id = torch.LongTensor([self.lang_id_map.get(lang, 0)])
        voc_in.speaker_id = torch.LongTensor([self.speaker_id_map.get(speaker_name, 0)])
        voc_in.to(self.device)
        return self.evaluate(voc_in, opt)

    @check_path(assert_file_exists=True)
    def resynthesize(
        self,
        wav_path: tp_PATH,
        ref_wav_path: tp.Optional[tp_PATH] = None,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: VocoderOptions = VocoderOptions(),
    ) -> VocoderForwardOutput:
        audio_chunk = (
            AudioChunk(file_path=wav_path).load(sr=self.sample_rate).volume(1.25)
        )
        ds = TTSDataSample(audio_chunk=audio_chunk)
        batch = self.pipe.datasample_to_batch([ds])
        collated: SpectrogramCollateOutput = batch.collated_samples  # type: ignore

        if ref_wav_path is not None:
            ref_audio_chunk = AudioChunk(file_path=ref_wav_path).load(sr=self.sample_rate)
            ref_ds = TTSDataSample(audio_chunk=ref_audio_chunk)
            ref_batch = self.pipe_for_reference.datasample_to_batch([ref_ds])
            ref_collated: SpectrogramCollateOutput = ref_batch.collated_samples  # type: ignore
            collated.speaker_emb = ref_collated.speaker_emb
            collated.speaker_emb_mean = ref_collated.speaker_emb_mean
            collated.spectrogram = ref_collated.spectrogram
            collated.spectrogram_lengths = ref_collated.spectrogram_lengths
            collated.averages = ref_collated.averages
            collated.speech_quality_emb = ref_collated.speech_quality_emb
            collated.additional_fields = ref_collated.additional_fields

        if self.model.__class__.__name__ == "Vocos":
            collated.speech_quality_emb = collated.speech_quality_emb * 0 + 5

            _input = VocoderForwardInput(
                spectrogram=collated.spectrogram,
                spectrogram_lengths=collated.spectrogram_lengths,
                ssl_feat=collated.ssl_feat,
                ssl_feat_lengths=collated.ssl_feat_lengths,
                plbert_feat=collated.plbert_feat,
                plbert_feat_lengths=collated.plbert_feat_lengths,
                speaker_emb=collated.speaker_emb,
                speaker_emb_mean=collated.speaker_emb,
                speech_quality_emb=collated.speech_quality_emb,
                averages=collated.averages,
                additional_inputs=collated.additional_fields,
                # energy=collated.energy,
                # pitch=collated.pitch,
            )

            if lang is not None:
                _input.lang_id = torch.LongTensor([self.lang_id_map[lang]])
            if speaker_name is not None:
                _input.speaker_id = torch.LongTensor([self.speaker_id_map[speaker_name]])

            _output = self.evaluate(_input, opt)
        else:
            raise NotImplementedError

        return _output


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    test_file_path = get_root_dir() / "tests/data/test_audio.wav"

    voc = VocoderEvaluationInterface(
        ckpt_path="mel_vocos_checkpoint_epoch=79_step=400000_val_loss=4.9470.ckpt"
    )

    voc_out = voc.resynthesize(test_file_path, lang="RU")
    voc_out.audio_chunk.save("resynt.wav", overwrite=True)
