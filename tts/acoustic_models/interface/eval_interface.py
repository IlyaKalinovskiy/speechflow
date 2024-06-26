import json
import pickle
import typing as tp
import logging

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from os import environ as env
from pathlib import Path

import numpy as np
import torch
import multilingual_text_parser

from multilingual_text_parser import Doc, Sentence, Syntagma, TextParser, Token

import speechflow

from nlp.prosody_prediction.eval_interface import ProsodyPredictionInterface
from speechflow.data_pipeline.core.components import PipelineComponents
from speechflow.data_pipeline.datasample_processors import add_pauses_from_text
from speechflow.data_pipeline.datasample_processors.biometric_processors import (
    VoiceBiometricProcessor,
)
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.text_processors import TextProcessor
from speechflow.io import check_path, tp_PATH
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.init import init_class_from_config, init_method_from_config
from speechflow.utils.seed import set_all_seed
from speechflow.utils.versioning import version_check
from tts import acoustic_models
from tts.acoustic_models.data_types import (
    TTSForwardInput,
    TTSForwardInputWithSSML,
    TTSForwardOutput,
)
from tts.acoustic_models.models.prosody_reference import (
    REFERENECE_TYPE,
    ComplexProsodyReference,
)

__all__ = [
    "TTSEvaluationInterface",
    "TTSContext",
    "TTSOptions",
]

LOGGER = logging.getLogger("root")

DEFAULT_SIL_TOKENS_NUM = 2


@dataclass
class TTSContext:
    prosody_reference: ComplexProsodyReference
    default_ds: tp.Optional[TTSDataSample] = None
    embeddings: tp.Optional[tp.Dict] = None
    additional_inputs: tp.Optional[tp.Dict] = None
    seed: int = 0
    sdk_version: str = speechflow.__version__

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = {}
        if self.additional_inputs is None:
            self.additional_inputs = {}

    @staticmethod
    def create(
        speaker_name: tp.Union[str, tp.Dict[str, str]],
        speaker_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
        style_reference: tp.Optional[
            tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]
        ] = None,
        seed: int = 0,
    ) -> "TTSContext":
        prosody_reference = ComplexProsodyReference.create(
            speaker_name,
            speaker_reference,
            style_reference,
        )
        return TTSContext(prosody_reference=prosody_reference, seed=seed)

    def copy(self) -> "TTSContext":
        return deepcopy(self)


@dataclass
class TTSOptions:
    gate_threshold: float = 0.2
    sigma_multiplier: float = 0.0
    average_val: tp.Optional[tp.Dict[str, float]] = None
    begin_pause: tp.Optional[float] = None
    end_pause: tp.Optional[float] = None
    forward_pass: bool = False
    speech_quality: float = 5.0
    tres_bin: float = 0.5
    predict_proba: bool = True
    use_spec_enhancement: bool = False

    def __post_init__(self):
        if self.average_val is None:
            self.average_val = {}

        self.average_val.setdefault("duration", 0.0)
        self.average_val.setdefault("energy", 0.0)
        self.average_val.setdefault("pitch", 0.0)
        self.average_val.setdefault("rate", 0.0)
        self.average_val.setdefault("duration_scale", 1.0)
        self.average_val.setdefault("energy_scale", 1.0)
        self.average_val.setdefault("pitch_scale", 1.0)
        self.average_val.setdefault("rate_scale", 1.0)

    def copy(self) -> "TTSOptions":
        return deepcopy(self)


class TTSEvaluationInterface:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        tts_ckpt_path: tp_PATH,
        prosody_ckpt_path: tp.Optional[tp_PATH] = None,
        pauses_ckpt_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
    ):
        env["DEVICE"] = device

        tts_ckpt = ExperimentSaver.load_checkpoint(tts_ckpt_path)
        data_cfg, cfg_model = ExperimentSaver.load_configs_from_checkpoint(tts_ckpt)

        version_check(
            multilingual_text_parser, tts_ckpt["versions"]["libs"]["text_parser"]
        )
        version_check(speechflow, tts_ckpt["versions"]["speechflow"])

        self.info = tts_ckpt.get("info", {})
        if not self.info:
            self.info = self._load_info(tts_ckpt_path)

        self.lang_id_map = tts_ckpt.get("lang_id_map", {})
        self.speaker_id_map = tts_ckpt.get("speaker_id_map", {})
        self.device = torch.device(device)

        # init model
        model_cls = getattr(acoustic_models, cfg_model["model"]["type"])
        self.model = model_cls(tts_ckpt["params"])
        self.model.eval()
        self.model.load_state_dict(tts_ckpt["state_dict"], strict=True)
        self.model.to(self.device)

        # update data config
        data_cfg["processor"].pop("dump", None)

        pauses_from_ts_cfg = data_cfg["preproc"]["pipe_cfg"].get(
            "add_pauses_from_timestamps", {}
        )
        pause_step = pauses_from_ts_cfg.get("step", 0.05)

        data_cfg["preproc"]["pipe"].insert(0, "add_pauses_from_text")
        data_cfg["preproc"]["pipe_cfg"]["add_pauses_from_text"] = {
            "level": "syntagmas",
            "num_symbols": 1 if pauses_ckpt_path else DEFAULT_SIL_TOKENS_NUM,
            "pause_from_punct_map": {
                ",": "normal",
                "-": "weak",
                "—": "normal",
                ".": "strong",
            },
            "step": pause_step,
        }
        data_cfg["preproc"]["pipe"].append("add_prosody_modifiers")

        text_cfg = data_cfg["preproc"]["pipe_cfg"].get("text", {})
        text_cfg["add_service_tokens"] = True

        self.lang = text_cfg.get("lang", "RU")
        if "alphabet" in tts_ckpt:
            assert tts_ckpt["alphabet"] == TextProcessor(self.lang).alphabet
        else:
            assert (
                tts_ckpt["params"]["n_symbols"]
                == TextProcessor(lang=self.lang).alphabet_size
            )

        data_cfg["collate"]["type"] = (
            "TTSCollateWithSSML"
            if "TTSCollate" in data_cfg["collate"]["type"]
            else data_cfg["collate"]["type"]
        )

        # init singleton handlers

        singleton_handlers = data_cfg.section("singleton_handlers", mutable=True)

        self.speaker_id_setter = singleton_handlers.get("SpeakerIDSetter")
        self.speaker_id_setter["resume_from_checkpoint"] = None
        if self.speaker_id_setter and self.lang_id_map:
            self.speaker_id_setter["lang_id_map"] = [
                f"{key}:{value}" for key, value in self.lang_id_map.items()
            ]
            self._dump_to_file("temp/lang_id_map.json", self.lang_id_map)
        if self.speaker_id_setter and self.speaker_id_map:
            self.speaker_id_setter["speaker_id_map"] = [
                f"{key}:{value}" for key, value in self.speaker_id_map.items()
            ]
            self._dump_to_file("temp/speaker_id_map.json", self.speaker_id_map)

        self.stat_ranges = self.info["singleton_handlers"].get("StatisticsRange")
        if self.stat_ranges is not None:
            handler = singleton_handlers.get("StatisticsRange", {})
            handler["statistics_file"] = "temp/StatisticsRange_data.json"
            self._dump_to_file(handler["statistics_file"], self.stat_ranges.statistics)

        self.mean_bio_embs = self.info["singleton_handlers"].get("MeanBioEmbeddings")
        # TODO: remove
        self.mean_bio_embs.mean_bio_embeddings = {
            s: np.asarray([emb], dtype=np.float32)
            for s, emb in self.mean_bio_embs.mean_bio_embeddings.items()
        }

        if self.mean_bio_embs is not None:
            embs = self.mean_bio_embs.mean_bio_embeddings
            embs = {s: emb.tolist() for s, emb in embs.items()}
            handler = singleton_handlers.get("MeanBioEmbeddings", {})
            handler["mean_embeddings_file"] = "temp/MeanBioEmbeddings_data.json"
            self._dump_to_file(handler["mean_embeddings_file"], embs)

        self.dataset_stat = self.info["singleton_handlers"].get("DatasetStatistics")
        if self.dataset_stat is not None:
            self.dataset_stat.unpickle()
            self.bio_embs = self.dataset_stat.speaker_embedding
            self.audio_hours_per_speaker = {
                name: value.sum() / 3600
                for name, value in self.dataset_stat.wave_duration.items()
            }
        else:
            self.bio_embs = None
            self.audio_hours_per_speaker = None  # type: ignore

        # init averages
        model_averages = self.model.get_params().get("averages", {})
        self.averages = self._get_averages_by_speaker(
            self.stat_ranges.statistics, model_averages
        )

        # init data pipeline
        self.pipeline = PipelineComponents(data_cfg, "test")
        self.sample_rate = data_cfg.section("preproc").find_field(
            "sample_rate", default_value=24000
        )
        self.hop_len = data_cfg.section("preproc").find_field("hop_len")

        ignored_fields = {
            "word_timestamps",
            "phoneme_timestamps",
            "speaker_emb",
        }
        pipeline = self.pipeline.with_ignored_fields(ignored_data_fields=ignored_fields)

        self.text_pipe = pipeline.with_ignored_fields(
            ignored_metadata_fields={"sega"},
            ignored_data_fields={"audio_chunk"},
        ).with_ignored_handlers(
            ignored_data_handlers={
                "add_pauses_from_text",
            }
        )
        self.spectrogram_pipe = pipeline.with_ignored_fields(
            ignored_metadata_fields={"sega"},
            ignored_data_fields={
                "pitch",
                "sent",
                "gate",
            },
        ).with_ignored_handlers(
            ignored_data_handlers={
                "average_by_time",
                "mean_bio_embedding",
                "VoiceBiometricProcessor",
            }
        )

        if "voice_bio_wespeaker" in data_cfg["preproc"]["pipe"]:
            self.bio_proc = init_class_from_config(
                VoiceBiometricProcessor,
                data_cfg["preproc"]["pipe_cfg"].voice_bio_wespeaker,
            )()
        else:
            self.bio_proc = None

        self.add_pauses_from_text = init_method_from_config(
            add_pauses_from_text, data_cfg["preproc"]["pipe_cfg"]["add_pauses_from_text"]
        )
        self.text_parser = {}

        # init batch processor
        cfg_model["batch"]["type"] = (
            "TTSBatchProcessorWithSSML"
            if cfg_model["batch"]["type"] == "TTSBatchProcessor"
            else cfg_model["batch"]["type"]
        )

        batch_processor_cls = getattr(acoustic_models, cfg_model["batch"]["type"])
        self.batch_processor = init_class_from_config(
            batch_processor_cls, cfg_model["batch"]
        )()
        self.batch_processor.device = self.device

        if prosody_ckpt_path is not None:
            if "_prosody" not in data_cfg["file_search"]["ext"]:
                LOGGER.warning("Current TTS model not support of prosody model!")
                self.prosody_ckpt_path = self.prosody_interface = None
            else:
                self.prosody_ckpt_path = Path(prosody_ckpt_path)
                self.prosody_interface = ProsodyPredictionInterface(
                    ckpt_path=self.prosody_ckpt_path,
                    lang=self.lang,
                    device=device,
                    text_parser=self.text_parser,
                )
        else:
            self.prosody_interface = None

        if pauses_ckpt_path is not None:
            self.pauses_interface = None
        else:
            self.pauses_interface = None

    @staticmethod
    def _load_info(ckpt_path: Path) -> tp.Dict[str, tp.Any]:
        for i in range(2):
            info_path = list(ckpt_path.parents[i].rglob("*info.pkl"))
            if info_path:
                print(f"Load info data from path {info_path[0].as_posix()}")
                with ExperimentSaver.portable_pathlib():
                    return pickle.loads(info_path[0].read_bytes())

        raise FileNotFoundError(f"*info.pkl file not found!")

    @staticmethod
    def _dump_to_file(file_name: str, data: tp.Any):
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        if Path(file_name).suffix == ".txt":
            Path(file_name).write_text(data, encoding="utf-8")
        elif Path(file_name).suffix == ".json":
            Path(file_name).write_text(json.dumps(data, indent=4), encoding="utf-8")
        elif Path(file_name).suffix == ".pkl":
            Path(file_name).write_bytes(pickle.dumps(data))
        else:
            raise NotImplementedError

    @staticmethod
    def _get_averages_by_speaker(
        stat_ranges: tp.Dict[str, tp.Any], model_averages: tp.Dict[str, tp.Any]
    ):
        averages: tp.Dict[str, tp.Dict[str, float]] = defaultdict(dict)

        for var in stat_ranges.keys():
            if var in model_averages:
                interval = model_averages[var]["interval"]
                averages[var]["default"] = np.array(interval).mean()
            else:
                averages[var]["default"] = np.array(0.5)

        if stat_ranges is not None:
            for var in stat_ranges.keys():
                for name, field in stat_ranges[var].items():
                    if field["max"] > 0.0:
                        if var != "rate":
                            averages[var][name] = field["mean"]  # / field["max"]
                        else:
                            averages[var][name] = field["mean"]
                    else:
                        averages[var][name] = averages[var]["default"]

        return averages

    def get_languages(self):
        return sorted(list(self.lang_id_map.keys()))

    def get_speakers(
        self,
        hours_per_speaker: tp.Optional[tp.Union[float, tp.Tuple[float, float]]] = None,
    ) -> tp.List[str]:
        if hours_per_speaker and self.dataset_stat:
            if isinstance(hours_per_speaker, float):
                names = [
                    name
                    for name, value in self.audio_hours_per_speaker.items()
                    if value > hours_per_speaker
                ]
            else:
                names = [
                    name
                    for name, value in self.audio_hours_per_speaker.items()
                    if hours_per_speaker[0] < value < hours_per_speaker[1]
                ]
        else:
            if self.audio_hours_per_speaker:
                names = list(self.audio_hours_per_speaker.keys())
            else:
                names = list(self.speaker_id_map.keys())

        return sorted(names)

    def predict_pauses(
        self,
        doc: Doc,
        begin_pause: tp.Optional[float],
        end_pause: tp.Optional[float],
        speaker_id: tp.Optional[int] = None,
    ):
        if self.pauses_interface is not None:
            pauses_output = self.pauses_interface.predict(
                doc,
                begin_pause=begin_pause,
                end_pause=end_pause,
                speaker_id=speaker_id,
            )
            pauses_durations = [
                pauses_output.durations[i][pauses_output.sil_masks[i] > 0]
                for i in range(pauses_output.durations.shape[0])
            ]
        else:
            pauses_durations = None

        doc.pauses_durations = (
            [None] * len(doc.sents) if pauses_durations is None else pauses_durations
        )
        return doc

    def prepare_text(
        self,
        text: str,
        lang: str,
        opt: tp.Optional[TTSOptions] = None,
    ) -> Doc:
        if self.lang_id_map and lang not in self.lang_id_map:
            raise ValueError(f"Language {lang} not support in current TTS model!")

        if lang not in self.text_parser:
            LOGGER.info(f"Initial TextParser for {lang} language")
            self.text_parser[lang] = TextParser(lang, device=str(self.device))

        if opt is None:
            opt = TTSOptions()

        doc = self.text_parser[lang].process(Doc(text))

        doc = self.predict_pauses(doc, opt.begin_pause, opt.end_pause)
        return doc

    def predict_prosody_by_text(self, doc: Doc, ctx: TTSContext, opt: TTSOptions) -> Doc:
        if self.prosody_interface is not None:
            doc = self.prosody_interface.predict(
                doc,
                tres_bin=opt.tres_bin,
                predict_proba=opt.predict_proba,
                seed=ctx.seed,
            )
        return doc

    def prepare_embeddings(
        self,
        lang: str,
        ctx: TTSContext,
        opt: TTSOptions,
    ) -> TTSContext:
        lang_id = self.lang_id_map.get(lang, 0)
        ctx.prosody_reference.initialize(
            self.speaker_id_map,
            self.bio_embs,
            self.mean_bio_embs.mean_bio_embeddings,
            self.bio_proc,
            self.spectrogram_pipe,
            seed=ctx.seed,
        )

        set_all_seed(ctx.seed)

        ctx.prosody_reference.set_feat_from_model(self.model)

        if ctx.prosody_reference.default.style_audio_path is not None:
            ds = TTSDataSample(
                audio_chunk=ctx.prosody_reference.default.style_audio_chunk
            )
            ref_ds = self.spectrogram_pipe.preprocessing_datasample([ds.copy()])[0]

        ds = TTSDataSample(
            lang_id=lang_id,
            mel=ctx.prosody_reference.default.style_spectrogram,
            speaker_name=ctx.prosody_reference.default.speaker_name,
            speaker_emb=ctx.prosody_reference.default.speaker_emb,
            speaker_emb_mean=ctx.prosody_reference.default.speaker_emb_mean,
            speech_quality_emb=torch.FloatTensor([[opt.speech_quality] * 4]),
        )

        ds.averages = {}
        for key in self.averages.keys():
            value = self.averages[key].get(ds.speaker_name)  # type: ignore
            if value is None or value == 0.0:
                value = 20 if key == "rate" else 0.0
            if opt.average_val.get(key, 0.0) == 0.0:
                ds.averages[key] = value
            else:
                ds.averages[key] = deepcopy(opt.average_val[key])

            scale = opt.average_val.get(f"{key}_scale", 1.0)
            ds.averages[key] *= 4 * scale

        ds.ranges = {}
        if self.stat_ranges is not None:
            for attr in self.stat_ranges.get_keys():
                f0_min, f0_max = self.stat_ranges.get_range(attr, ds.speaker_name)  # type: ignore
                ds.ranges[attr] = np.asarray(
                    [f0_min, f0_max, f0_max - f0_min], dtype=np.float32
                )

        batch = self.spectrogram_pipe.to_batch([ds])
        model_inputs, _, _ = self.batch_processor(batch)

        with torch.no_grad():
            output = self.model.embedding_component(model_inputs)

        embeddings = {
            k: v.cpu().numpy() for k, v in output.embeddings.items() if v is not None
        }

        ctx.default_ds = ds
        ctx.embeddings = embeddings
        return ctx

    def split_sentences(
        self,
        doc: Doc,
        max_sentence_length: tp.Optional[int] = None,
        max_text_length_in_batch: tp.Optional[int] = None,
        one_sentence_per_batch: bool = False,
    ) -> tp.List[tp.List[Sentence]]:
        sents = []
        for sent in doc.sents:
            sent = self.add_pauses_from_text(TTSDataSample(sent=sent)).sent

            if max_sentence_length and sent.num_phonemes > max_sentence_length:
                pause = Token(TextProcessor.sil)
                pause.phonemes = (TextProcessor.sil,)
                new_tokens: tp.List[Token] = []
                total_sent_length = 0
                for token in sent.tokens + [None]:
                    if token and token.num_phonemes > max_sentence_length:
                        raise RuntimeError("Invalid text!")

                    if (
                        token is None
                        or total_sent_length + token.num_phonemes > max_sentence_length
                    ):
                        new_tokens = [pause] + new_tokens + [pause]
                        new_sent = deepcopy(sent)
                        new_sent.tokens = new_tokens
                        new_sent.syntagmas = [Syntagma(new_tokens)]
                        sents.append(new_sent)
                        new_tokens = [token]
                        total_sent_length = token.num_phonemes if token else 0
                    else:
                        new_tokens.append(token)
                        total_sent_length += token.num_phonemes
            else:
                sents.append(sent)

        sents_by_batch = [[sents[0]]]
        total_text_length = sents[0].num_phonemes
        for sent in sents[1:]:
            if one_sentence_per_batch or (
                max_text_length_in_batch
                and total_text_length + sent.num_phonemes > max_text_length_in_batch
            ):
                sents_by_batch.append([])
                total_text_length = 0

            sents_by_batch[-1].append(sent)
            total_text_length += sent.num_phonemes

        return sents_by_batch

    def prepare_batch(
        self,
        sents: tp.List[Sentence],
        ctx: TTSContext,
        opt: TTSOptions,
    ) -> TTSForwardInputWithSSML:
        samples = []
        for sent in sents:
            new_ds = ctx.default_ds.copy()
            new_ds.sent = sent
            samples.append(new_ds)

        batch = self.text_pipe.datasample_to_batch(samples, skip_corrupted_samples=False)

        additional_inputs = {}

        for name, field in ctx.additional_inputs.items():
            if field is not None and isinstance(field, np.ndarray):
                field = torch.as_tensor(field)
                field = field.expand((batch.size, field.shape[1], field.shape[2]))
                additional_inputs[name] = field.to(self.device)

        for k, v in ctx.embeddings.items():
            if isinstance(v, np.ndarray):
                additional_inputs[k] = torch.cat(
                    [torch.from_numpy(v).to(self.device)] * batch.size
                )
            elif isinstance(v, torch.Tensor):
                additional_inputs[k] = torch.cat([v] * batch.size)
            elif v is not None:
                additional_inputs[k] = v

        if opt.sigma_multiplier is not None:
            additional_inputs["sigma_multiplier"] = opt.sigma_multiplier

        inputs, _, _ = self.batch_processor(batch)

        inputs.additional_inputs.update(additional_inputs)
        inputs.prosody_reference = ctx.prosody_reference.copy()
        inputs.output_lengths = None
        return inputs

    @torch.inference_mode()
    def evaluate(
        self,
        inputs: TTSForwardInputWithSSML,
        ctx: TTSContext,
        opt: TTSOptions,
    ) -> TTSForwardOutput:
        set_all_seed(ctx.seed)

        if opt.forward_pass:
            outputs = self.model(inputs)
        else:
            outputs = self.model.inference(inputs)
            if outputs.gate is not None and opt.gate_threshold is not None:
                outputs.output_mask = (
                    (outputs.gate.sigmoid() > opt.gate_threshold).cumsum(1) > 0
                ).squeeze(-1)

        setattr(outputs, "additional_inputs", inputs.additional_inputs)
        return outputs

    def synthesize(
        self,
        text: str,
        lang: str,
        speaker_name: str,
        ctx: tp.Optional[TTSContext] = None,
        opt: tp.Optional[TTSOptions] = None,
    ) -> tp.Tuple[TTSForwardOutput, TTSContext, TTSOptions]:
        if ctx is None:
            ctx = TTSContext.create(speaker_name)
        if opt is None:
            opt = TTSOptions()

        text_by_sentence = self.prepare_text(text, lang, opt)
        text_by_sentence = self.predict_prosody_by_text(text_by_sentence, ctx, opt)
        ctx = self.prepare_embeddings(lang, ctx, opt)
        inputs = self.prepare_batch(self.split_sentences(text_by_sentence)[0], ctx, opt)
        outputs = self.evaluate(inputs, opt)
        return outputs, ctx, opt

    def inference(self, batch_input: TTSForwardInput, **kwargs):
        return self.model.inference(batch_input, **kwargs)
