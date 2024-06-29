import typing as tp

from copy import deepcopy
from dataclasses import dataclass
from os import environ as env
from pathlib import Path

import torch

from tqdm import tqdm

from speechflow.data_pipeline.collate_functions.spectrogram_collate import (
    SpectrogramCollateOutput,
)
from speechflow.data_pipeline.core import PipelineComponents
from speechflow.data_pipeline.datasample_processors import SignalProcessor
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.io import AudioChunk, Config, check_path, tp_PATH
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from tts.vocoders.data_types import (
    VocoderForwardInput,
    VocoderForwardOutput,
    VocoderInferenceInput,
    VocoderInferenceOutput,
)
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
    ):
        env["DEVICE"] = device

        self.ckpt_path = ckpt_path

        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(self.ckpt_path)
        else:
            checkpoint = ckpt_preload

        self.data_cfg, model_cfg = ExperimentSaver.load_configs_from_checkpoint(
            checkpoint
        )
        self.data_cfg["collate"]["type"] = "SpectrogramCollate"

        self.pipe = self._load_data_pipeline(self.data_cfg)
        self.pipe_for_reference = self.pipe.with_ignored_handlers(
            ignored_data_handlers={"SSLProcessor"}
        )
        self.lang_id_map = checkpoint.get("lang_id_map", {})
        self.speaker_id_map = checkpoint.get("speaker_id_map", {})

        model_cfg["model"]["feature_extractor"]["init_args"].n_langs = len(
            self.lang_id_map
        )
        model_cfg["model"]["feature_extractor"]["init_args"].n_speakers = len(
            self.speaker_id_map
        )

        # Load model
        if self._check_vocos_signature(checkpoint):
            self.model = self._load_vocos_model(model_cfg, checkpoint)
        else:
            raise NotImplementedError

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

    @staticmethod
    def _check_vocos_signature(checkpoint: tp.Dict) -> bool:
        return "Vocos" in checkpoint["files"]["model.yml"]

    @staticmethod
    def _load_data_pipeline(data_cfg: Config) -> PipelineComponents:
        data_cfg["processor"].pop("dump", None)
        data_cfg["singleton_handlers"]["handlers"] = []
        pipe = PipelineComponents(Config(data_cfg).trim("ml"), data_subset_name="test")
        return pipe.with_ignored_fields(
            ignored_data_fields={"sent", "phoneme_timestamps"}
        ).with_ignored_handlers(
            ignored_data_handlers={"SignalProcessor", "WaveAugProcessor", "normalize"}
        )

    @staticmethod
    def _load_vocos_model(model_cfg: Config, checkpoint: tp.Dict[str, tp.Any]) -> Vocos:
        model = Vocos.init_from_config(model_cfg["model"])
        model.eval()

        state_dict: tp.Dict[str, tp.Any] = checkpoint["state_dict"]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "discriminators" not in k and "loss" not in k
        }
        model.load_state_dict(state_dict)
        return model


class VocoderEvaluationInterface(VocoderLoader):
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        ckpt_path: tp_PATH,
        device: str = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
    ):
        super().__init__(ckpt_path, device, ckpt_preload)
        self.sample_rate = find_field(self.data_cfg, "sample_rate")
        self.hop_size = find_field(self.data_cfg, "hop_len")
        self.n_mels = find_field(self.data_cfg, "n_mels")
        self.preemphasis_coef = self.find_preemphasis_coef(self.data_cfg)

    @staticmethod
    def find_preemphasis_coef(data_cfg: Config):
        beta = None
        for item in data_cfg["preproc"]["pipe_cfg"].values():
            if isinstance(item, dict) and item.get("type", None) == "SignalProcessor":
                if "preemphasis" in item.get("pipe", []):
                    if item["pipe_cfg"].get("preemphasis"):
                        beta = item["pipe_cfg"]["preemphasis"].get("beta", 0.97)
                    else:
                        beta = 0.97
        return beta

    @torch.inference_mode()
    def evaluate(
        self, inputs: VocoderInferenceInput, opt: VocoderOptions
    ) -> VocoderInferenceOutput:
        inputs.to(self.device)
        output = self.model.inference(inputs)
        return output

    @staticmethod
    def inv_preemphasis(wave: torch.Tensor, beta: float = 0.97) -> torch.Tensor:
        audio_chunk = AudioChunk(data=wave.cpu().numpy(), sr=1)
        inv_wave = SignalProcessor.inv_preemphasis(
            AudioDataSample(audio_chunk=audio_chunk), beta=beta
        ).audio_chunk.waveform
        inv_wave = torch.from_numpy(inv_wave)
        return inv_wave

    def synthesize(
        self,
        voc_in: VocoderForwardInput,
        voc_out: VocoderForwardOutput,
        opt: tp.Optional[VocoderOptions] = None,
    ) -> tp.Union[VocoderForwardOutput, tp.List[VocoderForwardOutput]]:
        if opt is None:
            opt = VocoderOptions()

        voc_in = voc_in
        voc_in.lang_id = voc_in.lang_id * 0 + self.lang_id_map[opt.lang]
        voc_in.speaker_id = voc_in.speaker_id * 0 + self.speaker_id_map[opt.speaker_name]
        voc_in.mel_spectrogram = voc_out.after_postnet_spectrogram
        voc_in.spectrogram_lengths = voc_out.spectrogram_lengths
        voc_in.energy = voc_out.variance_predictions["energy"]
        voc_in.pitch = voc_out.variance_predictions["pitch"]

        voc_in.to(self.device)
        output = self.model(voc_in)
        return output

    @check_path(assert_file_exists=True)
    def resynthesize(
        self,
        wav_path: tp_PATH,
        ref_wav_path: tp.Optional[tp_PATH] = None,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        opt: tp.Optional[VocoderOptions] = None,
    ) -> VocoderInferenceOutput:
        if opt is None:
            opt = VocoderOptions()

        audio_chunk = (
            AudioChunk(file_path=wav_path).load(sr=self.sample_rate).volume(1.25)
        )
        ds = SpectrogramDataSample(audio_chunk=audio_chunk)
        batch = self.pipe.datasample_to_batch([ds])
        collated: SpectrogramCollateOutput = batch.collated_samples  # type: ignore

        if ref_wav_path is not None:
            ref_audio_chunk = AudioChunk(file_path=ref_wav_path).load(sr=self.sample_rate)
            ref_ds = SpectrogramDataSample(audio_chunk=ref_audio_chunk)
            ref_batch = self.pipe_for_reference.datasample_to_batch([ref_ds])
            ref_collated: SpectrogramCollateOutput = ref_batch.collated_samples  # type: ignore
            collated.speaker_emb = ref_collated.speaker_emb
            collated.averages = ref_collated.averages
            collated.speech_quality_emb = ref_collated.speech_quality_emb
            collated.additional_fields = ref_collated.additional_fields

        if self.model.__class__.__name__ == "Vocos":
            collated.averages["energy"] = collated.averages["energy"] * 0 + 25
            collated.averages["pitch"] = collated.averages["pitch"] * 0 + 115
            collated.speech_quality_emb = collated.speech_quality_emb * 0 + 5

            _input = VocoderInferenceInput(
                mel_spectrogram=collated.mel_spectrogram,
                spectrogram_lengths=collated.spectrogram_lengths,
                ssl_feat=collated.ssl_feat,
                ssl_feat_lengths=collated.ssl_feat_lengths,
                speaker_emb=collated.speaker_emb,
                # energy=collated.energy,
                # pitch=collated.pitch,
                speech_quality_emb=collated.speech_quality_emb,
                averages=collated.averages,
                additional_inputs=collated.additional_fields,
                input_lengths=collated.spectrogram_lengths,
                output_lengths=collated.spectrogram_lengths,
            )

            if lang is not None:
                _input.lang_id = torch.LongTensor([self.lang_id_map[lang]])
            if speaker_name is not None:
                _input.speaker_id = torch.LongTensor([self.speaker_id_map[speaker_name]])

            _output = self.evaluate(_input, opt)
        else:
            raise NotImplementedError

        if self.preemphasis_coef is not None:
            _output.waveform = self.inv_preemphasis(
                _output.waveform, self.preemphasis_coef
            )

        return _output


if __name__ == "__main__":
    if 1:
        from speechflow.utils.fs import get_root_dir

        test_file_path = get_root_dir() / "tests/data/test_audio.wav"

        voc = VocoderEvaluationInterface(
            ckpt_path="mel_vocos_checkpoint_epoch=79_step=400000_val_loss=4.9470.ckpt"
        )

        outputs = voc.resynthesize(test_file_path, lang="RU")

        waveform = outputs.waveform[0].cpu().squeeze(0).numpy()
        AudioChunk(data=waveform, sr=voc.sample_rate).save("resynt.wav", overwrite=True)
    else:
        voc_path = "vocos_checkpoint_epoch=6_step=1439522_val_loss=9.8176.pt"
        test_files = "eng_spontan_2mic/wav_16k"
        result_path = "eng_spontan_2mic/result"
        ref_file = "4922.wav"

        voc = VocoderEvaluationInterface(voc_path, device="cuda:0")
        Path(result_path).mkdir(parents=True, exist_ok=True)

        file_list = list(Path(test_files).glob("*.wav"))
        for wav_file in tqdm(file_list):
            result_file_name = Path(result_path) / wav_file.name
            if result_file_name.exists():
                continue

            try:
                outputs = voc.resynthesize(wav_file, ref_file, lang="EN")
                waveform = outputs.waveform[0].cpu().squeeze(0).numpy()
                AudioChunk(data=waveform, sr=voc.sample_rate).save(
                    result_file_name, overwrite=True
                )
            except Exception as e:
                print(e)
