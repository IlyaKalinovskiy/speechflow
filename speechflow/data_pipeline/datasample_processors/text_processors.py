import random
import typing as tp
import logging
import itertools

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from multilingual_text_parser import Doc, Sentence, TextParser, Token, TokenUtils
from multilingual_text_parser.utils.model_loaders import load_transformer_model
from transformers import AutoTokenizer

from speechflow.data_pipeline.core.base_ds_processor import BaseDSProcessor
from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.data_types import (
    PausesPredictionDataSample,
    TextDataSample,
)
from speechflow.io import AudioSeg
from speechflow.logging import trace
from speechflow.utils.fs import get_module_dir, get_root_dir
from speechflow.utils.init import lazy_initialization
from speechflow.utils.profiler import Profiler

__all__ = ["load_text_from_sega", "TextProcessor", "LMProcessor"]

LOGGER = logging.getLogger("root")


class ZeroSilTokensError(Exception):
    """Exception raised in TextProcessor if allow_zero_sil parameter is False."""

    pass


@PipeRegistry.registry(
    inputs={"file_path"}, outputs={"word_timestamps", "phoneme_timestamps"}
)
def load_text_from_sega(ds: TextDataSample):
    sega = AudioSeg.load(ds.file_path)

    sega.ts_bos = ds.audio_chunk.begin
    sega.ts_eos = ds.audio_chunk.end
    sega.audio_chunk.begin = ds.audio_chunk.begin
    sega.audio_chunk.end = ds.audio_chunk.end
    word_ts, phoneme_ts = sega.get_timestamps(relative=True)

    ds.sent = sega.sent
    ds.word_timestamps = word_ts
    ds.phoneme_timestamps = phoneme_ts
    return ds


class TextProcessor(BaseDSProcessor):
    # service tokens
    pad = "<PAD>"
    bos = "<BOS>"
    eos = "<EOS>"
    sil = "<SIL>"
    unk = "<UNK>"
    sntgm = "<SNTGM>"
    eosntgm = "<EOSNTGM>"
    tkn = "<TKN>"
    eotkn = "<EOTKN>"
    unkpos = "<UNK_POS>"
    unkpunct = "<UNK_PUNCT>"
    emphasis = "<EMPHSIS>"
    no_emphasis = "<NOEMPHSIS>"
    breath = "<BREATH>"
    no_breath = "<NOBREATH>"

    def __init__(
        self,
        lang: str,
        add_service_tokens: bool = False,
        allow_zero_sil: bool = True,
        token_level: bool = False,
        aggregate_syntagmas: bool = False,
        prob: float = 0.3,
        ignore_ling_feat: tp.Optional[tp.List[str]] = None,
    ):
        super().__init__()

        text_parser = TextParser(lang=lang, cfg={"pipe": []})

        self.lang = lang
        self.is_complex_phoneme_token = text_parser.is_complex_phonemes
        self.num_symbols_per_phoneme_token = text_parser.num_symbols_per_phoneme

        self.service_tokens = (self.pad, self.bos, self.eos, self.sil, self.unk)
        self.phoneme_tokens = text_parser.phonemes
        self.punctuation_tokens = text_parser.punctuation
        self.pos_tokens = text_parser.pos
        self.rel_tokens = text_parser.rel
        self.intonation_tokens = text_parser.intonation
        self.additional_tokens = (
            self.sntgm,
            self.eosntgm,
            self.tkn,
            self.eotkn,
            self.unkpos,
            self.unkpunct,
            self.emphasis,
            self.no_emphasis,
            self.breath,
            self.no_breath,
        )

        self.ru2ipa = text_parser.ru2ipa
        self.en2ipa = text_parser.en2ipa

        self.alphabet = self.service_tokens + self.phoneme_tokens
        self._expand_alphabet(self.punctuation_tokens)
        self._expand_alphabet(self.pos_tokens)
        self._expand_alphabet(self.rel_tokens)
        self._expand_alphabet(self.intonation_tokens)
        self._expand_alphabet(self.additional_tokens)
        self._expand_alphabet(tuple([i + 1 for i in range(10)] + [-1]))
        self._expand_alphabet(
            tuple([f"<{p}>{self.sil}" for p in self.punctuation_tokens])
        )

        # Mappings from symbol to numeric ID and vice versa:
        self._symbol_to_id = {s: i for i, s in enumerate(self.alphabet)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.alphabet)}

        self._add_service_tokens = add_service_tokens
        self._allow_zero_sil = allow_zero_sil
        self._token_level = token_level
        self._aggregate_syntagmas = aggregate_syntagmas
        self._prob = prob
        self._ignore_ling_feat = ignore_ling_feat

        self._float_features = ["syntax_importance", "breath_mask"]

    def _expand_alphabet(self, new_symbols: tp.Tuple[str, ...]):
        self.alphabet += new_symbols

    def to_symbol(self, id: int) -> str:
        return self._id_to_symbol[id]

    @property
    def alphabet_size(self) -> int:
        return len(self.alphabet)

    @staticmethod
    def is_service_symbol(symbol: str) -> bool:
        service_tokens = (
            TextProcessor.pad,
            TextProcessor.bos,
            TextProcessor.eos,
            TextProcessor.sil,
            TextProcessor.unk,
        )
        return any(t in symbol for t in service_tokens)

    @PipeRegistry.registry(
        inputs={"sent"}, outputs={"symbols", "sequences", "symb_pad_id"}
    )
    def process(self, ds: TextDataSample) -> TextDataSample:
        if self.lang != "MULTILANG" and ds.sent.lang != self.lang:
            raise RuntimeError(
                f"The TextParser does not match the sentence {ds.sent.lang} language."
            )

        ph_by_word = ds.sent.get_phonemes()
        symbols = list(itertools.chain.from_iterable(ph_by_word))

        if self.lang == "MULTILANG":
            symbols = self.phons2ipa(ds.sent.lang, symbols)

        symbols = tuple(symbols)

        if self._token_level:
            ling_feat, word_lens, synt_lens = self._process_token_level(ds)
        else:
            ling_feat, word_lens, synt_lens = self._process_phoneme_level(ds, symbols)

        if self._ignore_ling_feat is not None:
            for name in self._ignore_ling_feat:
                if name not in ling_feat:
                    raise KeyError(f"Linguistic feature '{name}' not found!")
                else:
                    ling_feat.pop(name)

        # encode features
        for key, field in ling_feat.items():
            if key == "transcription":
                if not self.is_complex_phoneme_token:
                    ling_feat[key] = self._symbols_to_sequence(field)
                else:
                    ling_feat[key] = []
                    for i in range(self.num_symbols_per_phoneme_token):
                        seq = []
                        for s in symbols:
                            if isinstance(s, tuple):
                                seq.append(s[i] if len(s) > i else TextProcessor.unk)
                            else:
                                seq.append(s if i == 0 else TextProcessor.unk)

                        ling_feat[key].append(self._symbols_to_sequence(seq))
            else:
                if (
                    key not in self._float_features
                ):  # numerical, doesn't need to be encoded
                    ling_feat[key] = self._symbols_to_sequence(field)

        symbols, ling_feat, word_lens, synt_lens, ds.sent = self._assign_service_tokens(
            symbols,
            ling_feat,
            word_lens,
            synt_lens,
            ds.sent,
        )

        # check all features have equal lengths
        for key, field in ling_feat.items():
            dtype = (
                np.float32
                if (key in self._float_features and key != "prosody")
                else np.int64
            )
            ling_feat[key] = np.asarray(field, dtype=dtype)
            if ling_feat[key].ndim == 2:
                ling_feat[key] = ling_feat[key].T
            assert (
                len(ling_feat["sil_mask"]) == ling_feat[key].shape[0]
            ), "length sequence is mismatch!"

        if self._token_level:
            ds.symbols = None
            ds.transcription = None
        else:
            ds.symbols = symbols
            ds.transcription = ling_feat.pop("transcription")

        ds.word_lengths = np.asarray(word_lens, dtype=np.int64)
        ds.synt_lengths = np.asarray(synt_lens, dtype=np.int64)
        assert ds.word_lengths.sum() == len(ds.symbols)
        assert ds.word_lengths.sum() == ds.synt_lengths.sum()

        ds.ling_feat = ling_feat
        ds.pad_symb_id = self._symbol_to_id[self.pad]
        ds.sil_symb_id = self._symbol_to_id[self.sil]

        if isinstance(ds, PausesPredictionDataSample):
            ds.sil_mask = ling_feat.get("sil_mask")  # type: ignore

        self._set_token_lengths(ds)
        return ds

    def _syntagmas_aggregator(self, synt_lens):
        new_lens = [synt_lens[0]]
        for synt_len in synt_lens[1:]:
            if random.random() < self._prob:
                new_lens[-1] += synt_len
            else:
                new_lens.append(synt_len)

        assert sum(new_lens) == sum(synt_lens)
        return new_lens

    @staticmethod
    def _intonation_model(sentence: Sentence) -> str:
        if "?" in sentence.text:
            intonation_type = "quest_type0"
        elif "!" in sentence.text:
            intonation_type = "excl_type"
        else:
            intonation_type = "dot_type"
        return intonation_type

    def _process_emphasis(self, sentence: Sentence, token_lens):
        emph_labels = []
        for synt in sentence.syntagmas:
            for token in TokenUtils.get_word_tokens(synt.tokens):
                if token.emphasis == "accent":
                    emph_labels.append(self.emphasis)
                else:
                    emph_labels.append(self.no_emphasis)

        return emph_labels

    def _process_breath(self, sentence: Sentence):
        phonemes = sentence.get_phonemes()
        meta = TokenUtils.get_attr(sentence.tokens, attr_names=["meta"])["meta"]
        breath_mask = []
        for idx, (m, ph) in enumerate(zip(meta, phonemes)):
            if (
                self.sil not in ph[0]
                or idx + 1 == len(phonemes)
                or phonemes[idx + 1][0] == self.eos
            ):
                breath_mask.append([-10.0] * len(ph))
            else:
                if "noise_level" in m:
                    breath_mask.append(m["noise_level"])
                else:
                    breath_mask.append([-3.0] * len(ph))

        return tuple(itertools.chain(*breath_mask))

    def _process_phoneme_level(self, ds, symbols):
        word_lens, synt_lens, token_lens, lens_per_postag = self._count_phoneme_lens(
            ds.sent
        )
        emph_labels = self._process_emphasis(ds.sent, token_lens)
        breath_mask = self._process_breath(ds.sent)

        rels, head_counts = self.get_syntax(ds)
        expanded_rels = self._assign_tags_to_phoneme(list(zip(rels, token_lens)))
        expanded_head_count = self._assign_tags_to_phoneme(
            list(zip(head_counts, token_lens))
        )

        prosody = self.get_prosody(ds)
        prosody = self._assign_tags_to_phoneme(list(zip(prosody, token_lens)))

        syntamas_ends = self._assign_ends_of_items(synt_lens, self.sntgm, self.eosntgm)
        token_ends = self._assign_ends_of_items(token_lens, self.tkn, self.eotkn)
        pos_tags = self._assign_tags_to_phoneme(lens_per_postag)
        punctuation = self._assign_punctuation_to_phoneme(ds.sent)
        sil_mask = np.array([self.sil if self.sil in s else self.pad for s in symbols])
        emphasis = self._assign_tags_to_phoneme(list(zip(emph_labels, token_lens)))
        expanded_intonation = [self._intonation_model(ds.sent)] * len(emphasis)

        if not self._allow_zero_sil and len(sil_mask[sil_mask == self.sil]) == 0:
            raise ZeroSilTokensError("No sil tokens in the sentence")

        ling_feat = {
            "sil_mask": sil_mask,
            "transcription": symbols,
            "token_ends": token_ends,
            "syntagma_ends": syntamas_ends,
            "pos_tags": pos_tags,
            "punctuation": punctuation,
            "emphasis": emphasis,
            "intonation": expanded_intonation,
            "syntax": expanded_rels,
            "syntax_importance": expanded_head_count,
            "breath_mask": breath_mask,
            "prosody": prosody,
        }

        if self._aggregate_syntagmas:
            word_lens = self._syntagmas_aggregator(word_lens)
            synt_lens = self._syntagmas_aggregator(synt_lens)

        if (
            hasattr(ds, "aggregated")
            and isinstance(ds.aggregated, tp.MutableMapping)
            and ds.aggregated.get("word_durations") is not None
        ):
            word_durations = ds.aggregated.get("word_durations")
            word_durations = self._assign_tags_to_phoneme(
                list(zip(word_durations, token_lens))
            )
            ds.aggregated["word_durations"] = np.asarray(word_durations)

        return ling_feat, word_lens, synt_lens

    def _process_token_level(self, ds):
        word_lens, synt_lens, lens_per_postag = self._count_token_lens(ds.sent)

        rels, head_counts = self.get_syntax(ds)

        prosody = self.get_prosody(ds)

        syntamas_ends = self._assign_ends_of_items(synt_lens, self.sntgm, self.eosntgm)
        pos_tags = self._assign_tags_to_phoneme(lens_per_postag)
        punctuation = self._assign_punctuation_to_token(ds.sent)
        sil_mask = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if self.sil in token.text:
                    sil_mask.append(self.sil)
                else:
                    sil_mask.append(self.pad)
        sil_mask = np.array(sil_mask)
        if not self._allow_zero_sil and self.sil not in sil_mask:
            raise ZeroSilTokensError("No sil tokens in the sentence")

        ling_feat = {
            "sil_mask": sil_mask,
            "syntagma_ends": syntamas_ends,
            "pos_tags": pos_tags,
            "punctuation": punctuation,
            "syntax": rels,
            "syntax_importance": head_counts,
            "prosody": prosody,
        }
        return ling_feat, word_lens, synt_lens

    def get_prosody(self, ds):
        """Extract prosody features."""
        prosody = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                prosody.append(
                    int(token.prosody) + 1
                    if hasattr(token, "prosody")
                    and token.prosody
                    and token.prosody not in ["undefined", "-1"]
                    and token.emphasis != "accent"
                    else -1
                )
        return prosody

    def get_syntax(self, ds):
        """Extract features from SLOVNET."""
        full_rels, head_ids = [], []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if token.rel:
                    full_rels.append(token.rel)
                    head_ids.append(token.head_id)
                else:
                    full_rels.append(self.unk)
                    head_ids.append("-1")

        head_counts = Counter()
        for token in ds.sent.tokens:
            if not token.is_punctuation and token.head_id:
                head_counts[token.head_id] += 1

        full_head_counts = []
        for token in ds.sent.tokens:
            if not token.is_punctuation:
                if token.id:
                    full_head_counts.append(head_counts[token.id])
                else:
                    full_head_counts.append(0)

        return full_rels, full_head_counts

    def _symbols_to_sequence(self, symbols: tp.List[str]) -> tp.List[int]:
        return [self._symbol_to_id[self._is_symbol_in_alphabet(s)] for s in symbols]

    def _is_symbol_in_alphabet(self, s: str) -> str:
        if not s:
            return self.unk

        if s not in self._symbol_to_id:
            LOGGER.warning(trace(self, message=f"symbol [{s}] not in alphabet!"))
            s_rep = s.split(":", 1)[0]
            if s_rep in self.rel_tokens:
                LOGGER.warning(
                    trace(self, message=f"symbol [{s}] replaced with [{s_rep}]!")
                )
                return s_rep
            else:
                return self.unk
        else:
            return s

    @staticmethod
    def _assign_tags_to_phoneme(lens_per_tag: tp.List[tuple]) -> tp.List:
        """For every phoneme assign a tag.

        Parameters:
        -----------
        lens_per_tag: list of tuples.
            Every tuple must contain two elements (tag, length) with integer
            length for every tag value.

        """

        res = [[tag] * length for tag, length in lens_per_tag]
        return list(itertools.chain.from_iterable(res))

    def _assign_punctuation_to_token(self, sentence: Sentence) -> tp.Tuple:
        """For every token assign corresponding punctuation symbol."""
        punc_level = []
        in_quote = False
        prev_punct = current_punct = self.unkpunct
        all_tokens = tuple(
            ["T" if not token.is_punctuation else token.text for token in sentence.tokens]
        )
        all_tokens = tuple(itertools.chain.from_iterable(all_tokens))
        for i, symbol in enumerate(all_tokens[::-1]):
            if symbol not in self.punctuation_tokens:
                punc_level.append(current_punct)
                if current_punct == "(":
                    # brackets do not spread intonation further after closure.
                    current_punct = self.unkpunct
            else:
                if symbol == '"':
                    if in_quote:
                        current_punct = prev_punct
                    else:
                        in_quote = True
                        prev_punct, current_punct = current_punct, symbol
                elif symbol == "(":
                    current_punct = symbol
                else:
                    prev_punct, current_punct = current_punct, symbol

        return tuple(punc_level[::-1])

    def _assign_punctuation_to_phoneme(self, sentence: Sentence) -> tp.Tuple:
        """For every phoneme assign corresponding punctuation symbol."""
        punc_level = []
        in_quote = False
        prev_punct = current_punct = self.unkpunct
        all_phonemes = tuple(
            [
                token.phonemes if not token.is_punctuation else token.text
                for token in sentence.tokens
            ]
        )
        all_phonemes = tuple(itertools.chain.from_iterable(all_phonemes))
        for i, symbol in enumerate(all_phonemes[::-1]):
            if symbol not in self.punctuation_tokens:
                punc_level.append(current_punct)
                if current_punct == "(":
                    # brackets do not spread intonation further after closure.
                    current_punct = self.unkpunct
            else:
                if symbol == '"':
                    if in_quote:
                        current_punct = prev_punct
                    else:
                        in_quote = True
                        prev_punct, current_punct = current_punct, symbol
                elif symbol == "(":
                    current_punct = symbol
                else:
                    prev_punct, current_punct = current_punct, symbol

        return tuple(punc_level[::-1])

    @staticmethod
    def _assign_ends_of_items(
        lens: tp.List[int], in_symbol: str, end_symbol: str
    ) -> tp.List[tp.Any]:
        """For every phoneme assign `end_symbol` if phoneme is in the end of an item, else
        `in_symbol`.

        Parameters
        ----------
        lens : array-like
            contains lengths of every item in a sequence.

        """
        res = [
            [in_symbol] * max((length - 1), 0) + ([end_symbol] if length > 0 else [])
            for length in lens
        ]
        return list(itertools.chain.from_iterable(res))

    def _count_token_lens(
        self, sentence: Sentence
    ) -> tp.Tuple[tp.List, tp.List, tp.List]:
        """Count token lengths per syntagma."""
        word_lens, synt_lens, lens_per_postag = [], [], []
        for synt in sentence.syntagmas:
            tkn_in_syntagm = 0
            for token in synt.tokens:
                if not token.is_punctuation:
                    tkn_in_syntagm += 1
                    if self.sil not in token.text:
                        pos = token.pos if token.pos in self.pos_tokens else self.unkpos
                        lens_per_postag.append((pos, 1))
                    else:
                        lens_per_postag.append((self.sil, 1))
                    word_lens.append(1)
            synt_lens.append(tkn_in_syntagm)
        return word_lens, synt_lens, lens_per_postag

    def _count_phoneme_lens(
        self, sentence: Sentence
    ) -> tp.Tuple[tp.List, tp.List, tp.List, tp.List]:
        """Count phonemes lengths per token, syntagma and pos-tag."""
        word_lens, synt_lens, token_lens, lens_per_postag = [], [], [], []
        for synt in sentence.syntagmas:
            ph_in_syntagm = 0
            for token in synt.tokens:
                if not token.is_punctuation:
                    ph_in_token = token.num_phonemes
                    token_lens.append(ph_in_token)
                    ph_in_syntagm += ph_in_token
                    if self.sil not in token.text:
                        pos = token.pos if token.pos in self.pos_tokens else self.unkpos
                        lens_per_postag.append((pos, ph_in_token))
                    else:
                        lens_per_postag.append((self.sil, ph_in_token))
                    word_lens.append(ph_in_token)
            synt_lens.append(ph_in_syntagm)
        return word_lens, synt_lens, token_lens, lens_per_postag

    def _assign_service_tokens(
        self,
        symbols,
        ling_feat,
        word_lens,
        synt_lens,
        sentence,
    ) -> tp.Tuple[tp.Tuple[str, ...], tp.Dict, tp.List[int], tp.List[int], Sentence]:
        bos_id = self._symbol_to_id[self.bos]
        eos_id = self._symbol_to_id[self.eos]

        if self._add_service_tokens:
            symbols = (self.bos,) + symbols + (self.eos,)
            for key, field in ling_feat.items():
                if isinstance(field[0], tp.List):
                    for i in range(len(field)):
                        ling_feat[key][i] = [bos_id] + field[i] + [eos_id]
                elif isinstance(field[0], int):
                    ling_feat[key] = [bos_id] + field + [eos_id]
                else:
                    ling_feat[key] = (-1,) + field + (-1,)

            word_lens = [1] + word_lens + [1]
            synt_lens[0] += 1
            synt_lens[-1] += 1

            bos_token = Token(TextProcessor.bos)
            bos_token.phonemes = (TextProcessor.bos,)
            eos_token = Token(TextProcessor.eos)
            eos_token.phonemes = (TextProcessor.eos,)
            sentence.tokens = [bos_token] + sentence.tokens + [eos_token]

            syntagmas = sentence.syntagmas
            syntagmas[0].tokens = [bos_token] + syntagmas[0].tokens
            syntagmas[-1].tokens = syntagmas[-1].tokens + [eos_token]
            sentence.syntagmas = syntagmas
        else:
            for key, field in ling_feat.items():
                if key == "prosody":
                    continue
                if symbols[0] == self.bos:
                    if isinstance(field[0], tp.List):
                        for i in range(len(field)):
                            ling_feat[key][i][0] = bos_id
                    elif isinstance(field[0], int):
                        ling_feat[key][0] = bos_id
                    else:
                        ling_feat[key] = (-1,) + ling_feat[key][1:]
                if symbols[-1] == self.eos:
                    if isinstance(field[0], tp.List):
                        for i in range(len(field)):
                            ling_feat[key][i][-1] = eos_id
                    elif isinstance(field[0], int):
                        ling_feat[key][-1] = eos_id
                    else:
                        ling_feat[key] = ling_feat[key][:-1] + (-1,)

        return symbols, ling_feat, word_lens, synt_lens, sentence

    def _set_token_lengths(self, ds: TextDataSample):
        if not hasattr(ds, "aggregated"):
            return

        invert = []
        for p in ds.word_lengths:
            invert += [1 / p] * p

        ds.aggregated = {} if ds.aggregated is None else ds.aggregated
        ds.aggregated["word_lengths"] = ds.word_lengths
        ds.aggregated["word_invert_lengths"] = np.array(invert, dtype=np.float32)

    def phons2ipa(self, lang: str, phonemes: tp.List[str]) -> tp.List[str]:
        ipa_map = None
        if lang == "RU":
            ipa_map = self.ru2ipa
        elif lang == "EN":
            ipa_map = self.en2ipa

        if ipa_map is not None:
            return [
                ipa_map[phoneme] if phoneme in ipa_map else phoneme
                for phoneme in phonemes
            ]
        else:
            return phonemes


class LMProcessor(BaseDSProcessor):
    def __init__(
        self,
        lang: str,
        device: str = "cpu",
        by_phonemes: bool = True,
        model_dir: str = "data/ru/homo_classifier",
        model_name: str = "ruRoBerta",
    ):
        super().__init__(device=device)

        self._lang = lang
        self._device = device

        self.service_tokens = (
            TextProcessor.pad,
            TextProcessor.bos,
            TextProcessor.eos,
            TextProcessor.sil,
            TextProcessor.unk,
        )

        if not Path(model_dir).is_absolute():
            model_dir = model_dir.replace("multilingual_text_parser/", "")
            model_dir = model_dir.replace(
                "libs/multilingual_text_parser/multilingual_text_parser/", ""
            )
            model_dir = get_module_dir("multilingual_text_parser") / model_dir

        self._model_dir = model_dir
        self._model_name = model_name
        self._by_phonemes = by_phonemes
        self._lm_model = None
        self._tokenizer = None

    def init(self):
        super().init()
        self._lm_model = load_transformer_model(
            self._model_dir / self._model_name, output_hidden_states=True
        )
        self._lm_model.to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_dir / "tokenizer",
            use_fast=True,
            add_prefix_space=True,
        )

    @PipeRegistry.registry(inputs={"sent"}, outputs={"lm_feat"})
    @lazy_initialization
    def process(self, ds: TextDataSample) -> TextDataSample:
        if self._lang != "MULTILANG" and ds.sent.lang != self._lang:
            raise RuntimeError(
                f"The LMProcessor does not match the sentence {ds.sent.lang} language."
            )

        word_lens = self._count_word_lens(ds.sent)
        embeddings = self._process_lm(ds.sent)
        assert len(embeddings) == len(word_lens)
        if self._by_phonemes:
            ds.lm_feat = TextProcessor._assign_tags_to_phoneme(
                list(zip(embeddings, word_lens))
            )
        else:
            ds.lm_feat = embeddings

        ds.lm_feat = torch.stack(ds.lm_feat)
        return ds.to_numpy()

    @staticmethod
    def _count_word_lens(sentence: Sentence) -> tp.List:
        word_lens = []
        for synt in sentence.syntagmas:
            for token in synt.tokens:
                if not token.is_punctuation:
                    word_lens.append(token.num_phonemes)

        return word_lens

    def _process_lm(self, sentence: Sentence):
        with torch.inference_mode():
            tokens = [
                (t.text if t.text not in self.service_tokens else "<unk>")
                for t in sentence.tokens
            ]
            inp = self._tokenizer(
                [tokens],
                return_tensors="pt",
                max_length=512,
                is_split_into_words=True,
                truncation=True,
                padding=True,
            )
            pred = self._lm_model(
                input_ids=inp["input_ids"].to(self._device),
                attention_mask=inp["attention_mask"].to(self._device),
            ).last_hidden_state[0]

        word_ids = inp.word_ids()
        prev = None
        for i, j in enumerate(word_ids):
            if j is not None and prev != j:
                sentence.tokens[j].meta["embeddings"] = pred[i]
            prev = j

        embeddings = [
            t.meta["embeddings"] for t in sentence.tokens if not t.is_punctuation
        ]
        return embeddings


if __name__ == "__main__":
    utterance = """

    Летом Минздрав #России сообщал, что срок действия QR-кода может быть сокращен.

    """

    parser = TextParser(lang="RU")
    with Profiler(format=Profiler.Format.ms):
        doc = parser.process(Doc(utterance))

    PAUSE_SYMB = "<SIL>"
    ph_seq = []
    for sent in doc.sents:
        for synt in sent.syntagmas:
            attr = TokenUtils.get_attr(synt.tokens, ["text", "phonemes"], with_punct=True)
            for idx, word in enumerate(attr["phonemes"]):
                if word is not None:
                    ph_seq += list(word)
                else:
                    ph_seq += [attr["text"][idx]]
            ph_seq += [PAUSE_SYMB]
        print("----")
        print(sent.text)
        print("----")
        print(sent.stress)
        print("----")
        for token in sent.tokens:
            if token.modifiers:
                print(token.text, token.modifiers)

    print(ph_seq)

    text_processor = TextProcessor(lang="RU")
    _ds = TextDataSample(sent=doc.sents[0])
    _ds = text_processor.process(_ds)
    print(_ds.transcription)

    lm = LMProcessor(lang="RU", device="cpu")
    _ds = lm.process(_ds)
    print(_ds.lm_feat.shape)
