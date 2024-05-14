import typing as tp
import logging
import tempfile

from pathlib import Path

import numpy as np

from multilingual_text_parser import Doc, TextParser

from annotator.align import Aligner, AlignStage
from speechflow.io import AudioChunk, AudioSeg, Timestamps
from speechflow.utils.fs import get_root_dir

LOGGER = logging.getLogger("root")


class AnnotatorEvaluationInterface:
    def __init__(
        self,
        ckpt_stage1: tp.Union[str, Path],
        ckpt_stage2: tp.Union[str, Path],
        device: str,
        ckpt_stage1_preload: tp.Optional[dict] = None,
        ckpt_stage2_preload: tp.Optional[dict] = None,
        use_reverse_mode: bool = False,
    ):
        self.use_reverse_mode = use_reverse_mode

        self.aligner_stage1 = Aligner(
            ckpt_path=ckpt_stage1,
            stage=AlignStage.stage1,
            device=device,
            ckpt_preload=ckpt_stage1_preload,
        )
        self.aligner_stage2 = Aligner(
            ckpt_path=ckpt_stage2,
            stage=AlignStage.stage2,
            device=device,
            ckpt_preload=ckpt_stage2_preload,
        )

        if self.use_reverse_mode:
            self.aligner_stage1_reverse = Aligner(
                ckpt_path=ckpt_stage1,
                stage=AlignStage.stage1,
                device=device,
                ckpt_preload=ckpt_stage1_preload,
                reverse_mode=True,
                model_preload=self.aligner_stage1.model,
            )
            self.aligner_stage2_reverse = Aligner(
                ckpt_path=ckpt_stage2,
                stage=AlignStage.stage2,
                device=device,
                ckpt_preload=ckpt_stage2_preload,
                reverse_mode=True,
                model_preload=self.aligner_stage2.model,
            )

        self.text_parser = {}

    @property
    def lang(self) -> str:
        return self.aligner_stage1.lang

    @staticmethod
    def _cat_sentences(text: str) -> str:
        sents = Doc(text, sentenize=True, tokenize=True).sents
        if len(sents) > 1:
            sents = [sent.tokens[:-1] for sent in sents[:-1]] + [sents[-1].tokens]
            sents = [" ".join([token.text for token in sent]) for sent in sents]
            return ", ".join(sents)
        else:
            return text

    @staticmethod
    def _fix_pauses(file_name: Path, min_pause_len) -> AudioSeg:
        file_name_reverse = file_name.with_suffix(f"{file_name.suffix}_reverse")

        sega = AudioSeg.load(file_name)
        _, ph_ts = sega.get_timestamps()

        sega_reverse = AudioSeg.load(file_name_reverse)
        _, ph_ts_reverse = sega_reverse.get_timestamps()

        for idx, (ts, ts_reverse) in enumerate(zip(ph_ts[:-1], ph_ts_reverse[:-1])):
            a = ts[-1][1]
            ar = ts_reverse[-1][1]
            b = ph_ts[idx + 1][0][0]
            br = ph_ts_reverse[idx + 1][0][0]
            if b - a < min_pause_len and br - ar < min_pause_len:
                ph_ts[idx][-1][1] = b

        sega.set_phoneme_timestamps(ph_ts, ts_begin=sega.ts_bos, ts_end=sega.ts_eos)
        return sega

    @staticmethod
    def _fix_last_word(file_name: Path):
        file_name_reverse = file_name.with_suffix(f"{file_name.suffix}_reverse")

        sega = AudioSeg.load(file_name)
        word_ts, ph_ts = sega.get_timestamps()

        sega_reverse = AudioSeg.load(file_name_reverse)
        word_ts_reverse, ph_ts_reverse = sega_reverse.get_timestamps()

        if (
            word_ts[-1][1] - word_ts[-1][0]
            < word_ts_reverse[-1][1] - word_ts_reverse[-1][0]
        ):
            a, b = ph_ts[-1][0][0], 0
            for idx, ts_reverse in enumerate(ph_ts_reverse[-1]):
                b = a + (ts_reverse[1] - ts_reverse[0])
                b = min(b, sega_reverse.ts_eos)
                ph_ts[-1][idx][0] = a
                ph_ts[-1][idx][1] = b
                a = b

            dura = np.diff(ph_ts[-1])
            if abs(dura[-1]) < 1.0e-4:
                dura -= dura * 0.01
                delta = (ph_ts[-1].duration - dura.sum()) / len(ph_ts[-1])
                dura += delta
                ph_ts[-1] = ph_ts[-1][0][0] + Timestamps.from_durations(dura)

            if sega_reverse.ts_eos - ph_ts[-1][-1][1] < 0.02:
                ph_ts[-1][-1][1] = sega_reverse.ts_eos

            ts_begin = sega.ts_bos
            ts_end = sega_reverse.ts_eos
            sega.set_phoneme_timestamps(ph_ts, ts_begin=ts_begin, ts_end=ts_end)

        return sega

    def prepare_text(
        self,
        text: str,
        lang: str,
    ) -> Doc:
        if (
            self.aligner_stage1.lang_id_map
            and lang not in self.aligner_stage1.lang_id_map
        ):
            raise ValueError(f"Language {lang} not support in current TTS model!")

        if lang not in self.text_parser:
            LOGGER.info(f"Initial TextParser for {lang} language")
            self.text_parser[lang] = TextParser(
                lang, device=str(self.aligner_stage1.device)
            )

        doc = self.text_parser[lang].process(Doc(text))
        return doc

    def get_sega_from_text(
        self,
        text: str,
        wav_path: Path,
        speaker_name: tp.Optional[str] = None,
        lang: tp.Optional[str] = None,
    ) -> AudioSeg:
        sents = self.prepare_text(self._cat_sentences(text), lang=lang).sents
        assert len(sents) == 1

        sent = sents[0]
        audio_chunk = AudioChunk(file_path=wav_path).load()

        words = sent.get_words()
        ts_intervals = np.linspace(audio_chunk.begin, audio_chunk.end, len(words) + 1)
        ts = Timestamps(np.asarray(list(zip(ts_intervals[:-1], ts_intervals[1:]))))

        sega = AudioSeg(audio_chunk, sent)
        sega.set_word_timestamps(ts)
        sega.meta["speaker_name"] = speaker_name
        sega.meta["lang"] = lang
        return sega

    def proccess(
        self,
        text: tp.Optional[str] = None,
        wav_path: tp.Optional[Path] = None,
        speaker_name: tp.Optional[str] = None,
        lang: tp.Optional[str] = None,
        sega_path: tp.Optional[Path] = None,
    ) -> AudioSeg:
        with tempfile.TemporaryDirectory() as tmp_dir:

            if sega_path is not None:
                sega = AudioSeg.load(sega_path)
                assert sega.meta.get("with_audio", False)
                wav_path = sega_path.with_suffix(".wav")
                sega.meta["wav_path"] = wav_path.as_posix()
            elif text is not None and wav_path is not None:
                sega = self.get_sega_from_text(text, wav_path, speaker_name, lang)
            else:
                raise NotImplementedError("Set 'text' and 'wav_path' or 'sega_path'")

            file_name = Path(tmp_dir) / f"{wav_path.name}.TextGrid"
            file_name = file_name.absolute()
            sega.save(file_name)

            self.aligner_stage1.align_sega(file_name)
            if self.use_reverse_mode:
                self.aligner_stage1_reverse.align_sega(file_name)

            file_name = file_name.with_suffix(".TextGridStage1")
            if self.use_reverse_mode:
                sega = self._fix_pauses(file_name, self.aligner_stage2.min_pause_len)
                sega.save(file_name)

            self.aligner_stage2.align_sega(file_name)
            if self.use_reverse_mode:
                self.aligner_stage2_reverse.align_sega(file_name)

            file_name = file_name.with_suffix(".TextGridStage2")
            if self.use_reverse_mode:
                sega = self._fix_last_word(file_name)
            else:
                sega = AudioSeg.load(file_name)

            if sega_path is not None:
                sega.meta["wav_path"] = wav_path.name

        return sega


if __name__ == "__main__":
    from annotator.audio_transcription import OpenAIASR

    glow_tts_stage1 = Path("P:\\cfm\\3\\epoch=19-step=208340.ckpt")
    glow_tts_stage2 = Path("P:\\cfm\\3\\epoch=29-step=312510.ckpt")

    _wav_path = get_root_dir() / "tests/data/test_audio.wav"
    _speaker_name = "Tatiana"

    _asr = OpenAIASR(lang="RU", model_name="tiny")
    _text = _asr.converter({"wav_path": _wav_path})[0]["text"]

    annotator = AnnotatorEvaluationInterface(
        glow_tts_stage1,
        glow_tts_stage2,
        device="cpu",
    )
    _sega = annotator.proccess(_text, _wav_path, _speaker_name, "RU")
    _sega.save("sega.TG", with_audio=True)
