import json
import pickle
import typing as tp
import dataclasses

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import multilingual_text_parser

from multilingual_text_parser import Doc, EmptyTextError, Sentence

import speechflow

from speechflow.io import AudioChunk, check_path, tp_PATH
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.profiler import Profiler
from speechflow.utils.seed import get_seed_from_string
from speechflow.utils.serialize import JsonEncoder
from tts.acoustic_models.interface.eval_interface import (
    TTSContext,
    TTSEvaluationInterface,
    TTSOptions,
)
from tts.acoustic_models.models.prosody_reference import REFERENECE_TYPE
from tts.vocoders.eval_interface import VocoderEvaluationInterface, VocoderOptions

__all__ = ["SpeechSynthesisInterface", "SynthesisOptions", "SynthesisOutput"]

tp_SP_OPTIONS = tp.Dict[str, "SynthesisOptions"]
tp_SP_STYLES = tp.Dict[str, tp.List["SynthesisContext"]]
tp_REF_TYPE = tp.Union[REFERENECE_TYPE, tp.Dict[str, REFERENECE_TYPE]]


@dataclass
class SynthesisOptions:
    tts: TTSOptions = None  # type: ignore
    vocoder: VocoderOptions = None  # type: ignore
    volume: float = 1.0
    output_sample_rate: int = 0
    use_gsm_preemphasis: bool = False

    def __post_init__(self):
        if self.tts is None:
            self.tts = TTSOptions()
        if self.vocoder is None:
            self.vocoder = VocoderOptions()

    @property
    def as_dict(self) -> tp.Dict[str, tp.Any]:
        obj_dict = deepcopy(dataclasses.asdict(self))
        return obj_dict

    @property
    def as_json(self) -> str:
        return json.dumps(self.as_dict, cls=JsonEncoder, ensure_ascii=False, indent=4)

    @staticmethod
    def init_from_dict(cfg: tp.Dict[str, tp.Any]) -> "SynthesisOptions":
        opt = SynthesisOptions()
        for key in dataclasses.asdict(opt.tts).keys():
            if key in cfg:
                setattr(opt.tts, key, cfg[key])
        for key in dataclasses.asdict(opt.vocoder).keys():
            if key in cfg:
                setattr(opt.vocoder, key, cfg[key])
        for key in dataclasses.asdict(opt).keys():
            if key in cfg:
                setattr(opt, key, cfg[key])
        return opt

    def set_value(self, name: str, value: tp.Union[str, float, int]):
        def convert_type(old_value, new_value):
            if isinstance(old_value, str):
                return str(new_value)
            elif isinstance(old_value, bool):
                if new_value in ["0", "False", "false"]:
                    return False
                else:
                    return True
            elif isinstance(old_value, int):
                return int(new_value)
            elif isinstance(old_value, float):
                return float(new_value)
            else:
                return new_value

        def find_field(d, key, name, value):
            if key == name:
                setattr(d, key, convert_type(getattr(d, key), value))
            else:
                field = getattr(d, key, [])
                if field is not None and isinstance(field, dict):
                    if name in field:
                        field[name] = convert_type(field[name], value)

        for key in dataclasses.asdict(self.tts).keys():
            find_field(self.tts, key, name, value)
        for key in dataclasses.asdict(self.vocoder).keys():
            find_field(self.vocoder, key, name, value)
        for key in dataclasses.asdict(self).keys():
            find_field(self, key, name, value)

    def copy(self) -> "SynthesisOptions":
        return deepcopy(self)


@dataclass
class SynthesisContext:
    tts: TTSContext

    @staticmethod
    def create(
        lang: str,
        speaker_name: tp.Union[str, tp.Dict[str, str]],
        speaker_reference: tp.Optional[tp_REF_TYPE] = None,
        style_reference: tp.Optional[tp_REF_TYPE] = None,
        seed: int = 0,
    ) -> "SynthesisContext":
        tts_ctx = TTSContext.create(
            lang, speaker_name, speaker_reference, style_reference, seed
        )
        return SynthesisContext(tts=tts_ctx)

    def copy(self) -> "SynthesisContext":
        return deepcopy(self)


@dataclass
class SynthesisOutput:
    audio_chunk: AudioChunk
    text: str = None  # type: ignore
    doc: Doc = None  # type: ignore
    ctx: tp.Optional[SynthesisContext] = None
    opt: tp.Optional[SynthesisOptions] = None
    label: tp.Optional[str] = None

    @property
    def as_dict(self) -> tp.Dict[str, tp.Any]:
        obj_dict = deepcopy(dataclasses.asdict(self))
        return obj_dict

    @property
    def as_json(self) -> str:
        return json.dumps(self.as_dict, cls=JsonEncoder, ensure_ascii=False, indent=4)


class SpeechSynthesisInterface:
    """Интерфейс для синтеза речи.

    Parameters
    ----------
    tts_ckpt_path: путь к чекпоинту TTS
    vocoder_ckpt_path: путь к чекпоинту вокодера
    speaker_options_path: путь к параметрам голосов (SynthesisOptions) [опционально]
    speaker_styles_path: путь к пресетам стилей [опционально]
    device: выбор устройства для инференса моделей

    Если указан metafile_path, то задание других аргументов не требуется.

    """

    @check_path(assert_file_exists=True)
    def __init__(
        self,
        tts_ckpt_path: tp.Optional[tp_PATH] = None,
        vocoder_ckpt_path: tp.Optional[tp_PATH] = None,
        prosody_ckpt_path: tp.Optional[tp_PATH] = None,
        speaker_options_path: tp.Optional[tp_PATH] = None,
        speaker_styles_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
        with_profiler: bool = False,
    ):
        self.speaker_options: tp.Optional[tp_SP_OPTIONS] = None
        self.speaker_styles: tp.Optional[tp_SP_STYLES] = None

        if speaker_options_path is not None:
            with ExperimentSaver.portable_pathlib():
                self.speaker_options = pickle.loads(
                    Path(speaker_options_path).read_bytes()
                )
        if speaker_styles_path is not None:
            with ExperimentSaver.portable_pathlib():
                self.speaker_styles = pickle.loads(Path(speaker_styles_path).read_bytes())

        self.tts = TTSEvaluationInterface(
            tts_ckpt_path=tts_ckpt_path,
            prosody_ckpt_path=prosody_ckpt_path,
            device=device,
        )
        self.vocoder = VocoderEvaluationInterface(
            ckpt_path=vocoder_ckpt_path, device=device
        )
        self.profiler = Profiler(enable=with_profiler, format=Profiler.format.ms)  # type: ignore

    @property
    def languages(self) -> tp.List[str]:
        return self.tts.get_languages()

    @property
    def speakers(self) -> tp.List[str]:
        return self.tts.get_speakers()

    @property
    def version(self) -> str:
        ver = [
            f"speech sdk: {speechflow.__version__}",
            f"text parser: {multilingual_text_parser.__version__}",
        ]
        try:
            import tts_models

            ver.append(f"tts models: {tts_models.__version__}")
        except Exception as e:
            print(e)

        return ", ".join(ver)

    def _get_context(
        self,
        lang: str,
        speaker_name: str,
        speaker_reference: tp.Optional[tp_REF_TYPE] = None,
        style_reference: tp.Optional[tp_REF_TYPE] = None,
        ctx: tp.Optional[SynthesisContext] = None,
    ):
        if ctx is None:
            if self.speaker_styles is not None and self.speaker_styles.get(speaker_name):
                ctx = self.speaker_styles[speaker_name][0].ctx
            else:
                ctx = SynthesisContext.create(
                    lang, speaker_name, speaker_reference, style_reference
                )

        return ctx.copy()

    def _get_options(self, speaker_name: str, opt: tp.Optional[SynthesisOptions] = None):
        if opt is None:
            if self.speaker_options is not None and self.speaker_options.get(
                speaker_name
            ):
                opt = self.speaker_options[speaker_name]
            else:
                opt = SynthesisOptions()

        return opt.copy()

    @staticmethod
    def _prepare_synthesis_context(
        ctx: SynthesisContext, ssml_tags: dict
    ) -> SynthesisContext:
        for pos, tag in ssml_tags:
            if pos != -1:
                continue
            try:
                if "seed" in tag:
                    ctx.tts.seed = get_seed_from_string(tag["seed"]["value"])
                if "style" in tag:
                    ref_tag = tag["style"].get("tag", "default")
                    ref_id = int(tag["style"].get("id", 0))
                    sp_name = tag["style"].get(
                        "speaker", ctx.tts.prosody_reference.default.speaker_name
                    )
                    new_ref = ctx.tts.prosody_reference.get(ref_tag)
                    new_ref.speaker_name = sp_name
                    new_ref.set_speaker_reference((sp_name, ref_id))
                    ctx.tts.prosody_reference.is_initialize = False
            except Exception as e:
                print(e)

        return ctx

    @staticmethod
    def _prepare_synthesis_options(
        opt: SynthesisOptions, ssml_tags: dict
    ) -> SynthesisOptions:
        for pos, tag in ssml_tags:
            if pos != -1:
                continue
            try:
                if "volume" in tag:
                    opt.set_value("volume", float(tag["volume"]["value"]) / 100)
                if "prosody" in tag:
                    if "pitch" in tag["prosody"]:
                        opt.set_value("pitch_scale", float(tag["prosody"]["pitch"]) / 100)
                    if "rate" in tag["prosody"]:
                        opt.set_value("rate_scale", float(tag["prosody"]["rate"]) / 100)
                if "options" in tag:
                    for opt_name, opt_val in tag["options"].items():
                        opt.set_value(opt_name, opt_val)
            except Exception as e:
                print(e)

        return opt

    @staticmethod
    def _postprocessing(
        audio_chunk: AudioChunk,
        opt: SynthesisOptions,
    ) -> AudioChunk:
        if opt.volume != 1.0:
            volume = max(0.0, min(4.0, opt.volume))
            audio_chunk.volume(volume, inplace=True)

        if opt.output_sample_rate:
            output_sample_rate = max(8000, min(48000, opt.output_sample_rate))
            audio_chunk.resample(output_sample_rate, inplace=True)

        if opt.use_gsm_preemphasis:
            audio_chunk.gsm_preemphasis(inplace=True)

        return audio_chunk

    def _evaluate(
        self,
        sents: tp.List[Sentence],
        ctx: SynthesisContext,
        opt: SynthesisOptions,
    ) -> tp.Tuple[AudioChunk, SynthesisContext]:
        tts_in = self.tts.prepare_batch(
            sents,
            ctx.tts,
            opt.tts,
        )
        self.profiler.tick("prepare batch")

        pass
        self.profiler.tick("prepare tts input")

        tts_out = self.tts.evaluate(tts_in, ctx.tts, opt.tts)
        self.profiler.tick("tts evaluate")
        # _plot_1d(tts_out.variance_predictions["pitch"].cpu().numpy())

        voc_out = self.vocoder.synthesize(
            tts_in,
            tts_out,
            lang=ctx.tts.prosody_reference.default.lang,
            speaker_name=ctx.tts.prosody_reference.default.speaker_name,
            opt=opt.vocoder,
        )
        self.profiler.tick("vocoder evaluate")

        audio_chunk = self._postprocessing(voc_out.audio_chunk, opt)
        self.profiler.tick("prepare output")
        return audio_chunk, ctx

    def prepare_text(
        self,
        text: str,
        lang: tp.Optional[str] = None,
        speaker_name: tp.Optional[str] = None,
        speaker_reference: tp.Optional[tp_REF_TYPE] = None,
        style_reference: tp.Optional[tp_REF_TYPE] = None,
        ctx: tp.Optional[SynthesisContext] = None,
        opt: tp.Optional[SynthesisOptions] = None,
        max_text_length: tp.Optional[int] = None,
        one_sentence_per_batch: bool = True,
    ) -> tp.Tuple[tp.List[tp.List[Sentence]], SynthesisContext, SynthesisOptions]:

        if lang:
            if lang not in self.tts.lang_id_map:
                raise ValueError(f"Unknown language '{lang}'!")
        else:
            lang = ctx.tts.prosody_reference.default.lang

        if speaker_name:
            if speaker_name not in self.tts.speaker_id_map:
                raise ValueError(f"Unknown speaker name '{speaker_name}'!")
        else:
            speaker_name = ctx.tts.prosody_reference.default.speaker_name

        ctx = self._get_context(
            lang, speaker_name, speaker_reference, style_reference, ctx=ctx
        )
        opt = self._get_options(speaker_name, opt=opt)

        if text.strip() == "<tts_version/>":
            text = self.version.replace(".", " ")

        try:
            doc = self.tts.prepare_text(
                text,
                lang,
                opt.tts,
            )
        except EmptyTextError:
            raise RuntimeError("Input text is empty!")

        ssml_tags = doc.sents[0].ssml_insertions

        ctx = self._prepare_synthesis_context(ctx, ssml_tags)
        opt = self._prepare_synthesis_options(opt, ssml_tags)

        doc = self.tts.predict_prosody_by_text(doc, ctx.tts, opt.tts)

        ctx.tts = self.tts.prepare_embeddings(
            ctx.tts,
            opt.tts,
        )

        sents_by_batch = self.tts.split_sentences(
            doc,
            max_sentence_length=max_text_length,
            max_text_length_in_batch=max_text_length,
            one_sentence_per_batch=one_sentence_per_batch,
        )

        if not sents_by_batch:
            RuntimeError("batch is empty")

        return sents_by_batch, ctx, opt

    def batch_to_audio(
        self,
        batch: tp.List[Sentence],
        ctx: SynthesisContext,
        opt: SynthesisOptions,
    ) -> tp.Tuple[AudioChunk, SynthesisContext]:
        audio_chunk, ctx = self._evaluate(batch, ctx, opt)
        return audio_chunk, ctx

    def synthesize(
        self,
        text: str,
        lang: str,
        speaker_name: str,
        speaker_reference: tp.Optional[tp_REF_TYPE] = None,
        style_reference: tp.Optional[tp_REF_TYPE] = None,
        ctx: tp.Optional[SynthesisContext] = None,
        opt: tp.Optional[SynthesisOptions] = None,
        max_text_length: tp.Optional[int] = None,
        one_sentence_per_batch: bool = True,
    ) -> SynthesisOutput:
        """Синтез аудио по тексту.

        Parameters
        ----------
        text: произвольный текст для синтеза
        lang: язык текста для синтеза
        speaker_name: имя диктора (получить список доступных имен можно через свойство 'speaker_names')
        speaker_reference: референсное аудио для вычисления эмбеддинга диктора [опционально]
        style_reference: референсное аудио для вычисления стиля [опционально]
        ctx: контекст синтезв [опционально]
        opt: настройки синтеза [опционально]
        max_text_length: максимальное количество символов в батче
        one_sentence_per_batch: строго одно предложение в батче

        """
        self.profiler.reset()

        if speaker_name not in self.tts.speaker_id_map:
            raise ValueError(f"Unknown speaker name '{speaker_name}'!")

        ctx = self._get_context(
            lang,
            speaker_name,
            speaker_reference,
            style_reference,
            ctx,
        )
        opt = self._get_options(speaker_name, opt=opt)
        self.profiler.tick("prepare input")

        if text.strip() == "<tts_version/>":
            text = self.version.replace(".", " ")

        try:
            doc = self.tts.prepare_text(
                text,
                lang,
                opt.tts,
            )
            self.profiler.tick("prepare text")
        except EmptyTextError:
            return SynthesisOutput(
                audio_chunk=AudioChunk.silence(0.1, self.vocoder.sample_rate)
            )

        ssml_tags = doc.sents[0].ssml_insertions

        ctx = self._prepare_synthesis_context(ctx, ssml_tags)
        self.profiler.tick("prepare synthesis context")

        opt = self._prepare_synthesis_options(opt, ssml_tags)
        self.profiler.tick("prepare synthesis options")

        doc = self.tts.predict_prosody_by_text(doc, ctx.tts, opt.tts)
        self.profiler.tick("predict prosody reference")

        ctx.tts = self.tts.prepare_embeddings(
            ctx.tts,
            opt.tts,
        )
        self.profiler.tick("prepare embeddings")

        sents_by_batch = self.tts.split_sentences(
            doc,
            max_sentence_length=max_text_length,
            max_text_length_in_batch=max_text_length,
            one_sentence_per_batch=one_sentence_per_batch,
        )
        self.profiler.tick("split sentences")

        if not sents_by_batch:
            RuntimeError("batch is empty")

        audios = []
        for sents in sents_by_batch:
            audio_chunk, ctx = self._evaluate(sents, ctx, opt)
            audios.append(audio_chunk)

        waveform = np.concatenate([item.waveform for item in audios])
        self.profiler.total_time("total")
        self.profiler.logging()

        return SynthesisOutput(
            audio_chunk=AudioChunk(data=waveform, sr=self.vocoder.sample_rate),
            text=text,
            doc=doc,
            ctx=ctx,
            opt=opt,
        )


if __name__ == "__main__":

    synt_interface = SpeechSynthesisInterface(
        tts_ckpt_path="C:\\SRS\\data\\cfm_tts\\epoch=14-step=62505.ckpt",
        vocoder_ckpt_path="C:\\SRS\\data\\cfm_tts\\vocos_checkpoint_epoch=3_step=100000_val_loss=8.2706.ckpt",
        prosody_ckpt_path=None,
        device="cpu",
        with_profiler=True,
    )

    for name in sorted(synt_interface.speakers):
        print(f"- {name}")

    _lang = "RU"
    _speaker_name = "Natasha"
    _style_reference = Path("C:\\SRS\\5.wav")

    synt_interface.synthesize(
        "прогрев",
        lang=_lang,
        speaker_name=_speaker_name,
        style_reference=_style_reference,
    )

    if 1:
        _utterances = """
        <options pitch_scale="1.2"><style id="-1" tag="decoder_speaker|postnet"/>
        Услышал, что на использование мобильного банка или некоторых операций наложено ограничение, а вы хотите его отменить.
        Уважаемый, я голосовой ассистент банка ОТП.
        Уважаемая, вы позвонили в банк ОТП.
        Я его виртуальный консультант.
        """

        for i in range(10):
            result = synt_interface.synthesize(
                _utterances,
                lang=_lang,
                speaker_name=_speaker_name,
                style_reference=_style_reference,
                max_text_length=500,
            )
            result.audio_chunk.save(f"tts_result_{i}.wav", overwrite=True)
    else:
        _utterances = [
            '<options pitch_scale="1.2"/> В Санкт-Петербурге в стадии проектирования находятся новые станции метрополитена.',
            '<intonation seed="23">Об этом 15 декабря 2022 года пишет газета «Петербургский дневник» со ссылкой на Смольный.</intonation>',
            "Разработанный, но пока еще не принятый Генплан Петербурга предусматривает появление 93-х новых объектов метрополитена.",
        ]
        # _utterances = [
        #     """
        #     <seed value="-1"/><style id="-1"/>
        #     Директор департамента финансовой стабильности ЦБ - Елизавета Данилова заявила, что в ноябре выдача льготной ипотеки, к примеру, сопоставима по объемам с октябрем, несмотря на действующие ограничения.
        #     В таких условиях ЦБ ничего не оставалось как ввести дестимулирующие меры. В документе регулятор пишет, что прибегнул к фактически запретительным мерам.
        #     Помимо роста доли закредитованных заемщиков рынок столкнулся с ценовым расслоением — разрыв цен на первичном и вторичном рынках недвижимости достиг 42%.
        #     Однако на фоне повышения ключевой ставки, ипотечное кредитование на вторичном рынке замедляется, что будет приводить к сокращению спроса и на первичном рынке.
        #     ЦБ не раз указывал на риски, связанные с перегревом рынка ипотечного кредитования, а также выступал с критикой льготных ипотечных программ, которые, по его мнению, «уместны только как антикризисная мера».
        #     Также именно с ипотекой регулятор связывал один из дисбалансов в экономике, так как «она накачана льготными и псевдольготными программами».
        #     В 2023 году Банк России всерьез взялся за охлаждение рынка ипотеки: регулятор повысил макронадбавки по ипотеке с низким первоначальным взносом и высокой долговой нагрузкой заемщиков.
        #     """
        # ]

        _first_utterance = _utterances[0]
        _audios = []

        # ------------------------   text normalize service   ------------------------
        # synt_interface = SpeechSynthesisInterface(...)

        _sents_by_batch, _synt_context, _synt_options = synt_interface.prepare_text(
            _first_utterance,
            _lang,
            _speaker_name,
            style_reference=_style_reference,
        )

        _batches = _sents_by_batch
        for _utter in _utterances[1:]:
            _sents_by_batch, _synt_context, _synt_options = synt_interface.prepare_text(
                _utter,
                ctx=_synt_context,
                opt=_synt_options,
            )
            _batches += _sents_by_batch

        # ------------------------   cache service   ------------------------
        Path("_batches.pkl").write_bytes(pickle.dumps(_batches))
        Path("_synt_context.pkl").write_bytes(pickle.dumps(_synt_context))
        Path("_synt_options.pkl").write_bytes(pickle.dumps(_synt_options))

        # ------------------------   tts service   ------------------------
        # synt_interface = SpeechSynthesisInterface(...)

        for i in range(10):
            result_path = Path(f"tts_result_{i}.wav")

            if 1:
                _batches = pickle.loads(Path("_batches.pkl").read_bytes())
                _synt_context = pickle.loads(Path("_synt_context.pkl").read_bytes())
                _synt_options = pickle.loads(Path("_synt_options.pkl").read_bytes())
            else:
                _batches, _synt_context, _synt_options = synt_interface.prepare_text(
                    _utterances[0],
                    style_reference=_style_reference,
                )

            _audios = []
            for _batch in _batches:
                _audio_chunk, _synt_context = synt_interface.batch_to_audio(
                    _batch, _synt_context, _synt_options
                )
                _audios.append(_audio_chunk)

            _waveform = np.concatenate([item.waveform for item in _audios])
            AudioChunk(data=_waveform, sr=_audios[0].sr).save(result_path, overwrite=True)
            print("DONE", result_path.as_posix())
