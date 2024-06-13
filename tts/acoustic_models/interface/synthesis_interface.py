import json
import pickle
import typing as tp
import dataclasses

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import multilingual_text_parser

from multilingual_text_parser import Doc, EmptyTextError, Sentence

import speechflow

from speechflow.io import AudioChunk
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
from tts.vocoders.eval_interface import (
    VocoderEvaluationInterface,
    VocoderInferenceOutput,
    VocoderOptions,
)

__all__ = ["SpeechSynthesisInterface", "SynthesisOptions", "SynthesisOutput"]

tp_PATH = tp.Union[str, Path]
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
    use_speech_enhancement: bool = False

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
        speaker_name: tp.Union[str, tp.Dict[str, str]],
        speaker_reference: tp.Optional[tp_REF_TYPE] = None,
        style_reference: tp.Optional[tp_REF_TYPE] = None,
        seed: int = 0,
    ) -> "SynthesisContext":
        tts_ctx = TTSContext.create(
            speaker_name, speaker_reference, style_reference, seed
        )
        return SynthesisContext(tts=tts_ctx)

    def copy(self) -> "SynthesisContext":
        return deepcopy(self)


@dataclass
class SynthesisOutput:
    wave_chunk: AudioChunk
    text: str = None  # type: ignore
    doc: Doc = None  # type: ignore
    ctx: tp.Optional[SynthesisContext] = None
    opt: tp.Optional[SynthesisOptions] = None
    ckpt_path: tp.Optional[tp.Dict[str, Path]] = None
    label: tp.Optional[str] = None

    def __post_init__(self):
        if self.ckpt_path is None:
            self.ckpt_path = {}

    def __eq__(self, other: "SynthesisOutput"):
        if self.ckpt_path.get("tts") == other.ckpt_path.get("tts"):
            if self.ctx is not None:
                return self.ctx.tts.prosody_reference == other.ctx.tts.prosody_reference
            else:
                return True
        else:
            return False

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
    build_path: путь к сборке TTS [опционально]
    device: выбор устройства для инференса моделей

    Если указан metafile_path, то задание других аргументов не требуется.

    """

    def __init__(
        self,
        tts_ckpt_path: tp.Optional[tp_PATH] = None,
        vocoder_ckpt_path: tp.Optional[tp_PATH] = None,
        prosody_ckpt_path: tp.Optional[str] = None,
        speaker_options_path: tp.Optional[tp_PATH] = None,
        speaker_styles_path: tp.Optional[tp_PATH] = None,
        build_path: tp.Optional[tp_PATH] = None,
        device: str = "cpu",
        with_speech_enhancement: bool = False,
        with_profiler: bool = False,
    ):
        tts_ckpt_preload: tp.Optional[tp.Dict] = None
        vocoder_ckpt_preload: tp.Optional[tp.Dict] = None
        prosody_ckpt_preload: tp.Optional[tp.Dict] = None

        self.speaker_options: tp.Optional[tp_SP_OPTIONS] = None
        self.speaker_styles: tp.Optional[tp_SP_STYLES] = None

        if build_path is not None:
            metafile = ExperimentSaver.load_checkpoint(Path(build_path))
            tts_ckpt_path = metafile["tts_ckpt_path"]
            tts_ckpt_preload = metafile["tts_ckpt"]
            vocoder_ckpt_path = metafile["vocoder_ckpt_path"]
            vocoder_ckpt_preload = metafile["vocoder_ckpt"]
            prosody_ckpt_path = metafile.get("prosody_ckpt_path")
            prosody_ckpt_preload = metafile.get("prosody_ckpt")
            self.speaker_options = metafile.get("speaker_options")
            self.speaker_styles = metafile.get("speaker_styles")
        else:
            if isinstance(speaker_options_path, (str, Path)):
                with ExperimentSaver.portable_pathlib():
                    self.speaker_options = pickle.loads(
                        Path(speaker_options_path).read_bytes()
                    )
            if isinstance(speaker_styles_path, (str, Path)):
                with ExperimentSaver.portable_pathlib():
                    self.speaker_styles = pickle.loads(
                        Path(speaker_styles_path).read_bytes()
                    )

        assert tts_ckpt_path
        assert vocoder_ckpt_path

        self.tts = TTSEvaluationInterface(
            ckpt_path=tts_ckpt_path,
            device=device,
            with_ssml=True,
            ckpt_preload=tts_ckpt_preload,
            prosody_ckpt_path=prosody_ckpt_path,
            prosody_ckpt_preload=prosody_ckpt_preload,
        )
        self.vocoder = VocoderEvaluationInterface(
            ckpt_path=vocoder_ckpt_path, device=device, ckpt_preload=vocoder_ckpt_preload
        )
        self.profiler = Profiler(enable=with_profiler, format=Profiler.format.ms)  # type: ignore

        self.ckpt_path = {
            "tts": self.tts.ckpt_path,
            "vocoder": self.vocoder.ckpt_path,
            "pauses": self.tts.pauses_ckpt_path,
            "prosody": self.tts.prosody_ckpt_path,
        }

        if with_speech_enhancement:
            from df.enhance import init_df

            self.df_model, self.df_state, _ = init_df()
        else:
            self.df_model = self.df_state = None

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
                    speaker_name, speaker_reference, style_reference
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

    def _prepare_output(
        self,
        output: tp.Union[VocoderInferenceOutput, tp.List[VocoderInferenceOutput]],
        sample_rate: int,
        opt: SynthesisOptions,
    ) -> AudioChunk:
        voc_out = output if isinstance(output, list) else [output]  # type: ignore
        # mel = voc_out[0].spectrogram.cpu().numpy()
        # _plot_spectrogram(mel[0].transpose())

        waveform = []
        for out in voc_out:
            waveform.append(out.waveform.cpu()[0])
        waveform = torch.cat(waveform)

        wave_chunk = AudioChunk(data=waveform.numpy(), sr=sample_rate)
        if opt.volume != 1.0:
            volume = max(0.0, min(4.0, opt.volume))
            wave_chunk.volume(volume, inplace=True)

        if opt.output_sample_rate:
            output_sample_rate = max(8000, min(48000, opt.output_sample_rate))
            wave_chunk.resample(output_sample_rate, inplace=True)

        if opt.use_gsm_preemphasis:
            wave_chunk.gsm_preemphasis(inplace=True)

        if opt.use_speech_enhancement and self.df_model is not None:
            from df.enhance import enhance

            audio = torch.FloatTensor(wave_chunk.resample(sr=self.df_state.sr()).wave)
            enhanced = enhance(self.df_model, self.df_state, audio.unsqueeze(0))
            wave_chunk = AudioChunk(data=enhanced.cpu().numpy()[0], sr=self.df_state.sr())

        return wave_chunk

    def _evaluate(
        self,
        sents: tp.List[Sentence],
        ctx: SynthesisContext,
        opt: SynthesisOptions,
    ) -> tp.Tuple[AudioChunk, SynthesisContext]:
        tts_input = self.tts.prepare_batch(
            sents,
            ctx.tts,
            opt.tts,
        )
        self.profiler.tick("prepare batch")

        pass
        self.profiler.tick("prepare tts input")

        tts_out = self.tts.evaluate(tts_input, ctx.tts, opt.tts)
        self.profiler.tick("tts evaluate")
        # _plot_1d(tts_out.variance_predictions["pitch"].cpu().numpy())

        voc_out = self.vocoder.synthesize(tts_out, opt.vocoder)
        self.profiler.tick("vocoder evaluate")

        wave_chunk = self._prepare_output(voc_out, self.vocoder.output_sample_rate, opt)
        self.profiler.tick("prepare output")
        return wave_chunk, ctx

    def prepare_text(
        self,
        text: str,
        lang: str,
        speaker_name: str,
        ctx: tp.Optional[SynthesisContext] = None,
        opt: tp.Optional[SynthesisOptions] = None,
        max_text_length: tp.Optional[int] = None,
        one_sentence_per_batch: bool = True,
    ) -> tp.Tuple[tp.List[tp.List[Sentence]], SynthesisContext, SynthesisOptions]:
        if speaker_name not in self.tts.speaker_id_map:
            raise ValueError(f"Unknown speaker name '{speaker_name}'!")

        ctx = self._get_context(speaker_name, ctx=ctx)
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
            lang,
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

    def batch_to_wave(
        self,
        batch: tp.List[Sentence],
        ctx: SynthesisContext,
        opt: SynthesisOptions,
    ) -> tp.Tuple[AudioChunk, SynthesisContext]:
        wave_chunk, ctx = self._evaluate(batch, ctx, opt)
        return wave_chunk, ctx

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
            speaker_name,
            speaker_reference,
            style_reference,
            ctx=ctx,
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
                wave_chunk=AudioChunk.silence(0.1, self.vocoder.output_sample_rate)
            )

        ssml_tags = doc.sents[0].ssml_insertions

        ctx = self._prepare_synthesis_context(ctx, ssml_tags)
        self.profiler.tick("prepare synthesis context")

        opt = self._prepare_synthesis_options(opt, ssml_tags)
        self.profiler.tick("prepare synthesis options")

        doc = self.tts.predict_prosody_by_text(doc, ctx.tts, opt.tts)
        self.profiler.tick("predict prosody reference")

        ctx.tts = self.tts.prepare_embeddings(
            lang,
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

        waves = []
        for sents in sents_by_batch:
            wave_chunk, ctx = self._evaluate(sents, ctx, opt)
            waves.append(wave_chunk)

        wave = np.concatenate([x.wave for x in waves])
        self.profiler.total_time("total")
        self.profiler.logging()

        return SynthesisOutput(
            wave_chunk=AudioChunk(data=wave, sr=waves[0].sr),
            text=text,
            doc=doc,
            ctx=ctx,
            opt=opt,
            ckpt_path=self.ckpt_path,
        )


if __name__ == "__main__":

    synt_interface = SpeechSynthesisInterface(
        tts_ckpt_path="P:\\cfm\\epoch=19-step=62500.ckpt",
        vocoder_ckpt_path="P:\\cfm\\vocos_checkpoint_epoch=121_step=1906372_val_loss=4.0908.ckpt",
        prosody_ckpt_path="P:\\cfm\\prosody_ru\\epoch=14-step=7034.ckpt",
        device="cpu",
        with_profiler=True,
    )

    for name in sorted(synt_interface.speakers):
        print(f"- {name}")

    synt_interface.synthesize("прогрев", "RU", speaker_name="Andrey")

    if 1:
        _lang = "RU"
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
                _lang,
                speaker_name="Vasily",
                max_text_length=500,
            )
            result.wave_chunk.save(f"P:/result_{i}.wav", overwrite=True)
    else:
        _lang = "RU"
        _speaker_name = "Tatiana_marketing"
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
        _waves = []

        # ------------------------   text normalize service   ------------------------
        # synt_interface = SpeechSynthesisInterface(...)

        _sents_by_batch, _synt_context, _synt_options = synt_interface.prepare_text(
            _first_utterance,
            _lang,
            _speaker_name,
        )

        _batches = _sents_by_batch
        for _utter in _utterances[1:]:
            _sents_by_batch, _synt_context, _synt_options = synt_interface.prepare_text(
                _utter,
                _lang,
                _speaker_name,
                _synt_context,
                _synt_options,
            )
            _batches += _sents_by_batch

        # ------------------------   cache service   ------------------------
        Path("P:/_batches.pkl").write_bytes(pickle.dumps(_batches))
        Path("P:/_synt_context.pkl").write_bytes(pickle.dumps(_synt_context))
        Path("P:/_synt_options.pkl").write_bytes(pickle.dumps(_synt_options))

        # ------------------------   tts service   ------------------------
        # synt_interface = SpeechSynthesisInterface(...)

        for i in range(10):
            result_path = Path(f"P:/Tatiana_marketing_new_voco_result_{i}.wav")

            if 1:
                _batches = pickle.loads(Path("P:/_batches.pkl").read_bytes())
                _synt_context = pickle.loads(Path("P:/_synt_context.pkl").read_bytes())
                _synt_options = pickle.loads(Path("P:/_synt_options.pkl").read_bytes())
            else:
                _batches, _synt_context, _synt_options = synt_interface.prepare_text(
                    _utterances[0],
                    _lang,
                    _speaker_name,
                )

            _waves = []
            for _batch in _batches:
                _wave, _synt_context = synt_interface.batch_to_wave(
                    _batch, _synt_context, _synt_options
                )
                _waves.append(_wave)

            _wave = np.concatenate([x.wave for x in _waves])
            AudioChunk(data=_wave, sr=_waves[0].sr).save(result_path, overwrite=True)
            print("DONE", result_path.as_posix())
