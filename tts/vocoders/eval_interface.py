import typing as tp

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from speechflow.data_pipeline.collate_functions.spectrogram_collate import (
    SpectrogramCollateOutput,
)
from speechflow.data_pipeline.core import PipelineComponents
from speechflow.data_pipeline.datasample_processors import MelProcessor, SignalProcessor
from speechflow.data_pipeline.datasample_processors.data_types import (
    AudioDataSample,
    SpectrogramDataSample,
)
from speechflow.io import AudioChunk
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from tts.acoustic_models.data_types import TTSForwardOutput
from tts.vocoders.data_types import VocoderForwardInput, VocoderInferenceOutput
from tts.vocoders.vocos.vocos.pretrained import Vocos

__all__ = ["VocoderEvaluationInterface", "VocoderOptions"]


@dataclass
class VocoderOptions:
    normalize: bool = False
    denormalize: bool = False
    batch_evaluation: bool = True
    batch_evaluation_type: str = "concatfast"

    def copy(self) -> "VocoderOptions":
        return deepcopy(self)


class VocoderLoader:
    def __init__(
        self,
        ckpt_path: tp.Union[str, Path],
        device: tp.Union[str, torch.device] = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
    ):
        self.ckpt_path = Path(ckpt_path)

        if ckpt_preload is None:
            checkpoint = ExperimentSaver.load_checkpoint(self.ckpt_path)
        else:
            checkpoint = ckpt_preload

        if "vocoder_ckpt" in checkpoint:
            checkpoint = checkpoint["vocoder_ckpt"]

        self.data_cfg = checkpoint["data_cfg"]

        self.pipe = self._load_data_pipeline(self.data_cfg)
        self.lang_id_map = checkpoint.get("lang_id_map", {})
        self.speaker_id_map = checkpoint.get("speaker_id_map", {})

        # Load model
        if self._check_vocos_signature(checkpoint):
            self.model = self._load_vocos_model(checkpoint)
        else:
            raise NotImplementedError

        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)

    @staticmethod
    def _check_vocos_signature(checkpoint: tp.Dict) -> bool:
        return "vocos" in str(checkpoint["config"])

    @staticmethod
    def _load_data_pipeline(data_cfg: tp.Dict) -> PipelineComponents:
        if "StatisticsRange" in data_cfg["singleton_handlers"]["handlers"]:
            data_cfg["singleton_handlers"]["handlers"].remove("StatisticsRange")
        if "dump" in data_cfg["processor"]:
            data_cfg["processor"].pop("dump")
        data_cfg["collate"]["type"] = "SpecCollate"
        pipe = PipelineComponents(data_cfg, data_subset_name="test")
        return pipe.with_ignored_fields(
            ignored_data_fields={"sent", "phoneme_timestamps"}
        ).with_ignored_handlers(
            ignored_data_handlers={"SignalProcessor", "WaveAugProcessor", "normalize"}
        )

    @staticmethod
    def _load_vocos_model(checkpoint: tp.Dict) -> Vocos:
        model_cfg = checkpoint["config"]
        model = Vocos.from_hparams("", config=model_cfg["model"]["init_args"])
        model.eval()

        state_dict: dict = checkpoint["state_dict"]
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if "discriminators" not in k and "loss" not in k
        }
        model.load_state_dict(state_dict)
        setattr(
            model.feature_extractor, "speaker_id_map", checkpoint.get("speaker_id_map")
        )

        return model


class VocoderEvaluationInterface(VocoderLoader):
    def __init__(
        self,
        ckpt_path: tp.Union[str, Path],
        device: str = "cpu",
        ckpt_preload: tp.Optional[dict] = None,
    ):
        super().__init__(ckpt_path, device, ckpt_preload)
        self.sample_rate = find_field(self.data_cfg, "sample_rate")
        self.hop_size = find_field(self.data_cfg, "hop_len")
        self.n_mels = find_field(self.data_cfg, "n_mels")
        self.preemphasis_coef = self.find_preemphasis_coef(self.data_cfg)

    @staticmethod
    def find_preemphasis_coef(data_cfg: tp.Dict):
        beta = None
        for item in data_cfg["preproc"].values():
            if isinstance(item, list):
                continue
            if item.get("type", None) == "SignalProcessor":
                if "preemphasis" in item.get("pipe", []):
                    if data_cfg["preproc"][item].get("preemphasis"):
                        beta = data_cfg["preproc"][item]["preemphasis"].get("beta", 0.97)
                    else:
                        beta = 0.97
        return beta

    @torch.inference_mode()
    def evaluate(
        self, input: VocoderForwardInput, opt: VocoderOptions
    ) -> VocoderInferenceOutput:
        input.to(self.device)
        output = self.model.generate(input)
        output.spectrogram = input.spectrogram
        return output

    def auto_batch_evaluate(
        self,
        spectrogram_batch: torch.Tensor,
        spectrogram_masks: torch.Tensor,
        opt: VocoderOptions,
    ) -> tp.List[VocoderInferenceOutput]:
        batch_size, spectrogram_length, channels = spectrogram_batch.shape

        vocoder_outputs = []

        mask_start_indexes = [
            spectrogram_length
            if spectrogram_masks is None
            else spectrogram_length - int(spectrogram_masks[i].sum())
            for i in range(batch_size)
        ]

        if opt.batch_evaluation_type == "concatfast":
            assert spectrogram_masks is not None
            input_spectrogram = spectrogram_batch[~spectrogram_masks.bool(), :].unsqueeze(
                0
            )
            vocoder_input = VocoderForwardInput(spectrogram=input_spectrogram)
            waveform = self.evaluate(vocoder_input, opt).waveform

            upsample_ratio = waveform.shape[1] // input_spectrogram.shape[1]
            for index in range(batch_size):
                start_index = sum(mask_start_indexes[:index]) * upsample_ratio
                end_index = start_index + mask_start_indexes[index] * upsample_ratio
                vocoder_outputs.append(
                    VocoderInferenceOutput(waveform=waveform[:, start_index:end_index])
                )
            return vocoder_outputs

        elif opt.batch_evaluation_type == "loop":
            for i in range(batch_size):
                spectrogram = spectrogram_batch[i, : mask_start_indexes[i], :].unsqueeze(
                    0
                )
                vocoder_input = VocoderForwardInput(
                    spectrogram=spectrogram,
                    spectrogram_lengths=mask_start_indexes[i],
                )
                vocoder_outputs.append(self.evaluate(vocoder_input, opt))

            return vocoder_outputs
        else:
            raise NotImplementedError(
                f"batch_evaluation_type={opt.batch_evaluation_type}"
            )

    @staticmethod
    def convert_spectrogram(spectrogram, opt: VocoderOptions):
        ds = SpectrogramDataSample(mel=spectrogram)
        backend = {"backend": "torchaudio"}

        if opt.normalize:
            ds = MelProcessor(backend).normalize(ds)

        if opt.denormalize:
            ds = MelProcessor(backend).denormalize(ds)

        return ds.mel

    @staticmethod
    def inv_preemphasis(wave: torch.Tensor, beta: float = 0.97) -> torch.Tensor:
        audio_chunk = AudioChunk(data=wave.cpu().numpy(), sr=1)
        inv_wave = SignalProcessor.inv_preemphasis(
            AudioDataSample(audio_chunk=audio_chunk), beta=beta
        ).audio_chunk.waveform
        inv_wave = torch.from_numpy(inv_wave)
        return inv_wave

    def synthesize(
        self, tts_output: TTSForwardOutput, opt: tp.Optional[VocoderOptions] = None
    ) -> tp.Union[VocoderInferenceOutput, tp.List[VocoderInferenceOutput]]:
        if opt is None:
            opt = VocoderOptions()

        spectrogram_to_take = (
            tts_output.after_postnet_spectrogram
            if getattr(tts_output, "after_postnet_spectrogram", None) is not None
            else tts_output.spectrogram
        )
        spectrogram_masks = tts_output.masks.get("spec")

        spectrogram_to_take = self.convert_spectrogram(spectrogram_to_take, opt)

        batch_size = spectrogram_to_take.shape[0]
        if opt.batch_evaluation and batch_size > 1:
            try:
                output = self.auto_batch_evaluate(
                    spectrogram_to_take, spectrogram_masks, opt
                )
            except Exception as e:
                print(f"{self.__class__.__name__}: {e}")
                opt = deepcopy(opt)
                opt.batch_evaluation_type = "loop"
                output = self.auto_batch_evaluate(
                    spectrogram_to_take, spectrogram_masks, opt
                )
        else:
            vocoder_input = VocoderForwardInput(
                spectrogram=spectrogram_to_take,
                spectrogram_lengths=tts_output.spectrogram_lengths,
                energy=tts_output.additional_content.get("energy_postprocessed"),
                pitch=tts_output.additional_content.get("pitch_postprocessed"),
                speaker_embedding=tts_output.additional_content.get("speaker_embedding"),
                additional_inputs=tts_output.additional_content,
            )
            output = self.evaluate(vocoder_input, opt)

        if self.preemphasis_coef is not None:
            if isinstance(output, list):
                for inference_output in output:
                    inference_output.waveform = self.inv_preemphasis(
                        inference_output.waveform, self.preemphasis_coef
                    )
            elif isinstance(output, VocoderInferenceOutput):
                output.waveform = self.inv_preemphasis(
                    output.waveform, self.preemphasis_coef
                )

        return output

    def resynthesize(
        self,
        wav_path: Path,
        ref_wav_path: tp.Optional[Path] = None,
        lang: tp.Optional[str] = None,
        opt: tp.Optional[VocoderOptions] = None,
    ) -> VocoderInferenceOutput:
        if opt is None:
            opt = VocoderOptions()

        audio_chunk = AudioChunk(file_path=wav_path).load(sr=self.sample_rate)
        ds = SpectrogramDataSample(audio_chunk=audio_chunk)
        ds = self.pipe.preprocessing_datasample([ds])[0]
        batch = self.pipe.datasample_to_batch([ds])
        collated: SpectrogramCollateOutput = batch.collated_samples  # type: ignore

        if ref_wav_path is not None:
            ref_audio_chunk = AudioChunk(file_path=ref_wav_path).load(sr=self.sample_rate)
            ref_ds = SpectrogramDataSample(audio_chunk=ref_audio_chunk)
            ref_ds = self.pipe.preprocessing_datasample([ref_ds])[0]
            ref_batch = self.pipe.datasample_to_batch([ref_ds])
            ref_collated: SpectrogramCollateOutput = ref_batch.collated_samples  # type: ignore
            collated.speaker_emb = ref_collated.speaker_emb
            collated.additional_fields = ref_collated.additional_fields

        if self.model.__class__.__name__ == "Vocos":
            _input = VocoderForwardInput(
                spectrogram=collated.mel_spectrogram,
                spectrogram_lengths=collated.spectrogram_lengths,
                ssl_embeddings=collated.ssl_feat,
                ssl_embeddings_lengths=collated.ssl_feat_lengths,
                speaker_embedding=collated.speaker_emb,
                energy=collated.energy,
                pitch=collated.pitch,
                additional_inputs=collated.additional_fields,
            )

            if lang is not None:
                _input.lang_id = torch.LongTensor([self.lang_id_map[lang]])

            _output = self.evaluate(_input, opt)
        else:
            raise NotImplementedError

        if self.preemphasis_coef is not None:
            _output.waveform = self.inv_preemphasis(
                _output.waveform, self.preemphasis_coef
            )

        return _output


if __name__ == "__main__":
    from speechflow.utils.fs import get_root_dir

    test_file_path = get_root_dir() / "tests/data/test_audio.wav"

    voc = VocoderEvaluationInterface(
        ckpt_path="P:\\cfm\\5\\mel_vocos_checkpoint_epoch=79_step=400000_val_loss=4.9470.ckpt"
    )

    outputs = voc.resynthesize(test_file_path, lang="RU")

    waveform = outputs.waveform[0].cpu().squeeze(0).numpy()
    AudioChunk(data=waveform, sr=voc.sample_rate).save("resynt.wav", overwrite=True)
