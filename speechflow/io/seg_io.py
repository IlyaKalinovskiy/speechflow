import json
import typing as tp
import itertools

from copy import copy
from pathlib import Path

import numpy as np
import more_itertools

from multilingual_text_parser import Doc, Position, Sentence, Syntagma, Token
from praatio import tgio
from praatio.tgio import Interval

from speechflow.io import AudioChunk, Timestamps
from speechflow.io.utils import check_path, tp_PATH

__all__ = ["AudioSeg"]


def _get_new_position(new_text: str, old_text: str):
    if old_text.startswith(new_text):
        return "first"
    elif old_text.endswith(new_text):
        return "last"
    else:
        return "internal"


class AudioSeg:
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        audio_chunk: AudioChunk,
        sent: Sentence,
        sega_path: tp.Optional[tp_PATH] = None,
    ):
        assert audio_chunk.duration > 0, "invalid wave!"
        assert len(sent) > 0, "sentence contains no tokens!"
        self.audio_chunk = audio_chunk
        self.sent: Sentence = sent
        self.ts_by_words: tp.Optional[Timestamps] = None
        self.ts_by_phonemes: tp.Optional[tp.List[Timestamps]] = None
        self.ts_bos: float = self.audio_chunk.begin
        self.ts_eos: float = self.audio_chunk.end
        self.meta: tp.Dict[str, tp.Any] = {}
        self.sega_path: tp.Optional[tp_PATH] = sega_path
        self.auxiliary: tp.Dict[str, tp.Any] = {}

    @staticmethod
    def _fp_eq(a, b, eps: float = 1.0e-6):
        return np.abs(np.float32(a) - np.float32(b)) < eps

    @staticmethod
    def _fix_meta_string(string: str) -> str:
        string = string.replace("{'", '{"').replace("'}", '"}')
        string = string.replace("', '", '", "')
        string = string.replace("': '", '": "')
        string = string.replace("': [", '": [').replace("], '", '], "')
        string = string.replace("': true, '", '": true, "')
        string = string.replace("': false, '", '": false, "')
        string = string.replace("\\'", "'")
        return string

    def set_word_timestamps(
        self,
        ts: Timestamps,
        ts_begin: tp.Optional[float] = None,
        ts_end: tp.Optional[float] = None,
        relative: bool = False,
    ):
        ts_begin = self.audio_chunk.begin if ts_begin is None else ts_begin
        ts_end = self.audio_chunk.end if ts_end is None else ts_end
        assert ts_begin is not None and ts_end is not None

        if relative:
            ts += self.audio_chunk.begin
            ts_begin += self.audio_chunk.begin
            ts_end += self.audio_chunk.begin

        self.ts_bos = ts_begin
        self.ts_eos = ts_end

        words = self.sent.get_words()
        self.ts_by_words = ts

        if not len(words) == len(ts):
            raise ValueError(
                f"Number of words: {len(words)} doesnt match number of passed timestamps: {len(ts)}"
            )
        if not np.float32(ts.begin) >= np.float32(self.audio_chunk.begin):
            raise ValueError(
                "Bounds of passed timestamps are not compatible with begin of wave."
            )
        if not np.float32(ts.end) <= np.float32(self.audio_chunk.end):
            raise ValueError(
                "Bounds of passed timestamps are not compatible with end of wave."
            )

        if self.ts_by_phonemes is None:
            self.ts_by_phonemes = []
            for (a, b), word in zip(ts, words):  # type: ignore
                if word.phonemes:
                    num_phonemes = len(word.phonemes)
                    step = float((b - a) / num_phonemes)
                    t = [(i * step, (i + 1) * step) for i in range(num_phonemes)]
                else:
                    t = [(0, b - a)]
                self.ts_by_phonemes.append(Timestamps(np.asarray(t)) + a)

    def set_phoneme_timestamps(
        self,
        ts: tp.Union[Timestamps, tp.List[Timestamps]],
        ts_begin: tp.Optional[float] = None,
        ts_end: tp.Optional[float] = None,
        relative: bool = False,
    ):
        if isinstance(ts, tp.List):
            ts = Timestamps(np.concatenate(ts))

        ts_by_phonemes = []
        for ph_word in self.sent.get_phonemes():
            ts_word, ts = ts[: len(ph_word)], ts[len(ph_word) :]
            ts_by_phonemes.append(Timestamps(np.asarray(ts_word)))
        assert (
            len(ts) == 0
        ), "The number of phonemes and the number of timestamps do not match."

        if relative:
            ts_by_phonemes = [ts + self.audio_chunk.begin for ts in ts_by_phonemes]

        self.ts_by_phonemes = ts_by_phonemes

        ts_by_words = [(ts.begin, ts.end) for ts in ts_by_phonemes]
        self.set_word_timestamps(Timestamps(np.asarray(ts_by_words)), ts_begin, ts_end)

    @property
    def duration(self) -> float:
        return self.ts_eos - self.ts_bos

    def get_tier(self, name: str, relative: bool = False):
        assert self.ts_by_words, "timestamps not set!"

        seq = []
        if name == "orig":
            seq.append(
                (self.ts_by_words.begin, self.ts_by_words.end, self.sent.text_orig)
            )
        elif name == "syntagmas":
            for synt in self.sent.syntagmas:
                word_begin = self.sent.get_word_index(synt[0])
                word_end = self.sent.get_word_index(synt[-1])
                ts_begin = self.ts_by_words[word_begin][0]
                ts_end = self.ts_by_words[word_end][1]
                seq.append((ts_begin, ts_end, synt.position.name))
        elif name == "phonemes":
            phonemes = self.sent.get_phonemes()
            for ts_list, ph_list in zip(self.ts_by_phonemes, phonemes):  # type: ignore
                if ph_list:
                    assert len(ts_list) == len(ph_list)
                    for ts, ph in zip(ts_list, ph_list):  # type: ignore
                        if not isinstance(ph, str):
                            ph = "|".join(ph)
                        seq.append((ts[0], ts[1], ph))
        elif name == "breath_mask":
            breath_mask = self.sent.get_attr(name, group=True, with_punct=name == "text")
            for ts_list, bm_list in zip(self.ts_by_phonemes, breath_mask):  # type: ignore
                if bm_list:
                    if isinstance(bm_list[0], tp.List):
                        bm_list = bm_list[0]
                    if len(ts_list) != len(bm_list):
                        bm_list = bm_list * len(ts_list)
                    for ts, m in zip(ts_list, bm_list):  # type: ignore
                        seq.append((ts[0], ts[1], "undefined" if m is None else str(m)))
        else:
            words = self.sent.get_attr(name, group=True, with_punct=name == "text")
            for ts, word in zip(self.ts_by_words, words):  # type: ignore
                word = [str(x) if x is not None else "undefined" for x in word]
                label = "".join(word)
                seq.append((ts[0], ts[1], label))

        if not self._fp_eq(self.audio_chunk.begin, self.ts_bos):
            seq.insert(0, (self.audio_chunk.begin, self.ts_bos, "BOS"))
        if not self._fp_eq(self.ts_eos, self.audio_chunk.end):
            seq.append((self.ts_eos, self.audio_chunk.end, "EOS"))

        if relative:
            offset = self.audio_chunk.begin
            for idx in range(len(seq)):
                begin, end, label = seq[idx]
                seq[idx] = (
                    max(begin - offset, 0),
                    min(end - offset, self.audio_chunk.duration),
                    label,
                )

        return seq

    def get_tier_for_meta(self, meta: dict, relative: bool = False):
        meta.update(self.meta)
        dump = json.dumps(meta, ensure_ascii=False).replace('"', "'")
        begin = self.audio_chunk.begin
        end = self.audio_chunk.end

        if relative:
            offset = self.audio_chunk.begin
            begin = max(begin - offset, 0)
            end = min(end - offset, self.audio_chunk.duration)

        tier = tgio.IntervalTier(
            "meta", [(begin, end, dump)], 0, maxT=self.audio_chunk.duration
        )
        return tier

    @staticmethod
    def _remove_service_tokens(
        tiers: dict,
    ) -> tp.Tuple[dict, tp.Optional[float], tp.Optional[float]]:
        ts_bos = ts_eos = None
        if tiers["text"][0][2] == "BOS":
            ts_bos = tiers["text"][0][1]
        if tiers["text"][-1][2] == "EOS":
            ts_eos = tiers["text"][-1][0]

        for name, field in tiers.items():
            tiers[name] = [item for item in field if item[2] not in ["BOS", "EOS"]]

        return tiers, ts_bos, ts_eos

    @staticmethod
    def timestamps_from_tier(
        tier: tp.List[tp.Tuple[float, float, str]]
    ) -> tp.Tuple["Timestamps", tp.Optional[float], tp.Optional[float]]:
        tiers, ts_begin, ts_end = AudioSeg._remove_service_tokens({"text": tier})
        tm = [t[:2] for t in tiers["text"]]
        return Timestamps(np.asarray(tm)), ts_begin, ts_end

    @check_path(make_dir=True)
    def save(
        self,
        file_path: tp_PATH,
        fields: tp.Optional[tp.List[str]] = None,
        with_audio: bool = False,
        with_meta: bool = True,
        overwrite: bool = True,
    ):
        if not overwrite and file_path.exists():
            raise RuntimeError(f"Sega {str(file_path)} is exists!")

        if fields is None:
            fields = [
                "orig",
                "syntagmas",
                "text",
                "stress",
                "pos",
                "phonemes",
                "emphasis",
                "id",
                "head_id",
                "rel",
                "asr_pause",
                "prosody",
                # "breath_mask",
            ]

        seqs = {}
        for name in fields:
            seqs[name] = self.get_tier(name, relative=with_audio)

        meta = {"lang": self.sent.lang, "sent_position": self.sent.position.name}

        tg = tgio.Textgrid()
        for name, field in seqs.items():
            tier = tgio.IntervalTier(name, field, 0, maxT=self.audio_chunk.duration)
            tg.addTier(tier)

        if with_audio:
            if self.audio_chunk.empty and self.audio_chunk.file_path:
                self.audio_chunk.load()

            self.audio_chunk.file_path = file_path.with_suffix(".wav")
            self.audio_chunk.save(overwrite=overwrite)
            meta["wav_path"] = self.audio_chunk.file_path.name
            meta["audio_chunk"] = (0.0, self.audio_chunk.duration)  # type: ignore
            meta["with_audio"] = True
        else:
            meta["wav_path"] = Path(self.audio_chunk.file_path).as_posix()
            meta["audio_chunk"] = (self.audio_chunk.begin, self.audio_chunk.end)  # type: ignore

        if with_meta:
            tier = self.get_tier_for_meta(meta, relative=with_audio)
            tg.addTier(tier)

        tg.save(file_path.as_posix())

    @classmethod
    def _get_sent(cls, tiers) -> Sentence:
        words = " ".join([word.label for word in tiers["text"]])
        doc = Doc(words, sentenize=True, tokenize=True)

        assert len(doc.sents) == 1, doc.text
        sent = doc.sents[0]

        if "orig" in tiers:
            sent.text_orig = tiers["orig"][0].label

        for token in sent.tokens:
            if token.is_punctuation:
                token.pos = "PUNCT"

        return sent

    @staticmethod
    @check_path(assert_file_exists=True)
    def load(
        file_path: tp_PATH,
        audio_path: tp.Optional[tp_PATH] = None,
        with_audio: bool = False,
        crop_begin: tp.Optional[float] = None,
        crop_end: tp.Optional[float] = None,
    ) -> "AudioSeg":
        tg = tgio.openTextgrid(file_path.as_posix())
        orig_text = " ".join(
            [
                t.label
                for t in tg.tierDict["text"].entryList
                if t.label not in ["BOS", "EOS"]
            ]
        )
        if crop_end is not None and crop_begin is not None:
            max_timestamp = tg.maxTimestamp
            tg = tg.crop(crop_begin, crop_end, rebaseToZero=False, mode="lax")
        else:
            max_timestamp = None

        tiers = {}
        for name, field in tg.tierDict.items():
            entrylist = field.entryList
            if (
                crop_end is not None
                and crop_begin is not None
                and name not in ("orig", "meta")
            ):
                entrylist.insert(0, Interval(start=0.0, end=crop_begin, label="BOS"))
                entrylist.append(Interval(start=crop_end, end=max_timestamp, label="EOS"))
            tiers[name] = entrylist

        if crop_end is not None and crop_begin is not None:
            new_text = " ".join(
                [t.label for t in tiers["text"] if t.label not in ["BOS", "EOS"]]
            )
            new_position = _get_new_position(new_text, tiers["orig"][0][2])
            tiers["orig"][0] = Interval(start=crop_begin, end=crop_end, label=new_text)

            new_meta = json.loads(AudioSeg._fix_meta_string(tiers["meta"][0].label))
            meta_bos_label, meta_eos_label = (
                new_meta.get("bos_label", ""),
                new_meta.get("eos_label", ""),
            )
            orig_part_bos = meta_bos_label + " " + orig_text[: orig_text.index(new_text)]
            orig_part_eos = (
                orig_text[orig_text.index(new_text) + len(new_text) :]
                + " "
                + meta_eos_label
            )

            new_meta["sent_position"] = new_position
            new_meta["bos_label"] = orig_part_bos.strip()
            new_meta["eos_label"] = orig_part_eos.strip()
            tiers["meta"][0] = Interval(
                start=crop_begin, end=crop_end, label=json.dumps(new_meta)
            )

        tiers, ts_bos, ts_eos = AudioSeg._remove_service_tokens(tiers)

        sent = AudioSeg._get_sent(tiers)
        ts_by_word = [(word[0], word[1]) for word in tiers["text"]]

        if "meta" in tiers:
            meta = json.loads(AudioSeg._fix_meta_string(tiers["meta"][0].label))
            orig_wav_path = Path(meta["wav_path"])
            sent.position = Position[meta["sent_position"]]
            sent.lang = meta.get("lang")
        else:
            meta = {}
            orig_wav_path = None  # type: ignore

        if orig_wav_path is None or not orig_wav_path.exists():
            orig_wav_path = file_path.with_suffix(".wav")

        audio_path = orig_wav_path if audio_path is None else audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file {audio_path.as_posix()} not found!")

        audio_chunk = AudioChunk(audio_path)

        if "audio_chunk" in meta:
            audio_chunk.begin = meta["audio_chunk"][0]
            audio_chunk.end = meta["audio_chunk"][1]

        if with_audio:
            audio_chunk.load()

            if meta.get("with_audio", False):
                assert AudioSeg._fp_eq(
                    audio_chunk.duration, meta["audio_chunk"][1], eps=1.0e-3
                )

        words = sent.get_words()
        assert len(words) == len(tiers["text"]), f"{file_path}"

        for name in [
            "stress",
            "pos",
            "emphasis",
            "id",
            "head_id",
            "rel",
            "asr_pause",
            "prosody",
        ]:
            if name in tiers:
                for (_, _, label), word in zip(tiers[name], words):
                    label = None if label == "undefined" else label
                    setattr(word, name, label)

        ph_word = []
        ts_by_ph = []
        word_idx = 0
        if "phonemes" in tiers:
            for begin, end, label in tiers["phonemes"]:
                if "|" in label:
                    label = tuple(label.split("|"))
                ph_word.append(label)
                ts_by_ph.append((begin, end))
                if AudioSeg._fp_eq(end, tiers["text"][word_idx][1]):
                    words[word_idx].phonemes = tuple(ph_word)
                    ph_word = []
                    word_idx += 1
            assert all([word.phonemes for word in words]), f"{file_path}"

        if "syntagmas" in tiers:
            synt_tokens: tp.List[tp.List[Token]] = []
            words_with_punct = sent.get_words_with_punct()
            for (begin, _, label), tokens in zip(tiers["text"], words_with_punct):
                if any([begin == synt[0] for synt in tiers["syntagmas"]]):
                    synt_tokens.append([])
                synt_tokens[-1] += tokens

            syntagmas = []
            for tokens, synt_position in zip(synt_tokens, tiers["syntagmas"]):
                syntagma = Syntagma(tokens)
                syntagma.position = Position[synt_position.label]
                syntagmas.append(syntagma)
            sent.syntagmas = syntagmas

        sega = AudioSeg(audio_chunk, sent, sega_path=file_path)
        if crop_end is not None and crop_begin is not None:
            sega.ts_bos = crop_begin
            sega.ts_eos = crop_end

        if ts_by_ph:
            sega.set_phoneme_timestamps(
                Timestamps(np.asarray(ts_by_ph)),
                ts_begin=ts_bos,
                ts_end=ts_eos,
            )
        else:
            sega.set_word_timestamps(
                Timestamps(np.asarray(ts_by_word)),
                ts_begin=ts_bos,
                ts_end=ts_eos,
            )

        sega.meta = meta
        return sega

    @staticmethod
    @check_path(assert_file_exists=True)
    def load_meta(file_path: tp_PATH) -> dict:
        line = file_path.read_text(encoding="utf-8")
        line = line[line.rfind("meta") + len("meta") + 1 :]
        line = line[line.find('"{') + 1 :][::-1]
        line = line[line.find('"}') + 1 :][::-1]
        return json.loads(AudioSeg._fix_meta_string(line))

    def get_timestamps(
        self, relative: bool = False
    ) -> tp.Tuple[Timestamps, tp.List[Timestamps]]:
        assert self.ts_by_words and self.ts_by_phonemes

        ts_by_words = copy(self.ts_by_words)
        ts_by_phonemes = copy(self.ts_by_phonemes)

        if relative:
            if ts_by_words is not None:
                ts_by_words -= self.audio_chunk.begin
            if ts_by_phonemes is not None:
                ts_by_phonemes = [ts - self.audio_chunk.begin for ts in ts_by_phonemes]

        return ts_by_words, ts_by_phonemes

    def split_into_syntagmas(self, min_offset: float = 0.1) -> tp.List["AudioSeg"]:
        syntagmas_timestamps = self.get_tier("syntagmas")[1:-1]
        part_idxs = {
            tuple(x)
            for x in itertools.chain.from_iterable(
                more_itertools.partitions(range(len(syntagmas_timestamps)))
            )
        }

        splitted_syntagmas = []
        for syntagma_position in part_idxs:

            if len(syntagma_position) == 2:
                idx_start, idx_end = syntagma_position
            elif len(syntagma_position) == 1:
                idx_start, idx_end = syntagma_position[0], syntagma_position[0]
            else:
                raise RuntimeError(
                    f"incorrect syntagmas timestamps in {self.sega_path} in splitting function."
                )

            syntagma_start = (
                syntagmas_timestamps[idx_start - 1][1] if idx_start != 0 else self.ts_bos
            )
            syntagma_end = (
                syntagmas_timestamps[idx_end + 1][0]
                if idx_end + 1 != len(syntagmas_timestamps)
                else self.ts_eos
            )

            if idx_start == 0:
                _bos_ts = self.ts_bos
                is_left_valid = True
            else:
                _bos_ts = syntagmas_timestamps[idx_start - 1][1]
                is_left_valid = syntagmas_timestamps[idx_start][0] - _bos_ts >= min_offset

            if idx_end == len(syntagmas_timestamps) - 1:
                _eos_ts = self.ts_eos
                is_right_valid = True
            else:
                _eos_ts = syntagmas_timestamps[idx_end + 1][0]
                is_right_valid = _eos_ts - syntagmas_timestamps[idx_end][1] >= min_offset

            if not (is_right_valid and is_left_valid):
                continue

            assert self.sega_path
            new_sega = self.load(
                self.sega_path,
                with_audio=True,
                crop_begin=syntagma_start,
                crop_end=syntagma_end,
            )
            new_sega.meta["split_idxs"] = "-".join([str(x) for x in syntagma_position])
            splitted_syntagmas.append(new_sega)

        return splitted_syntagmas
