import re
import math
import random
import typing as tp
import logging
import itertools

from copy import deepcopy as copy

import numpy as np

from annoy import AnnoyIndex
from multilingual_text_parser import Doc, Syntagma, TextParser, Token, TokenUtils
from scipy import interpolate, signal

from speechflow.data_pipeline.core.registry import PipeRegistry
from speechflow.data_pipeline.datasample_processors.algorithms.audio_processing.breath_detector import (
    detect_noise,
)
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.text_processors import TextProcessor
from speechflow.io import Timestamps
from speechflow.logging import trace

__all__ = [
    "add_pauses_from_text",
    "add_pauses_from_timestamps",
    "calc_durations",
    "calc_invert_durations",
    "add_gate_value",
    "add_service_tokens",
    "aggregate_by_phoneme",
    "curvature_estimate_by_phoneme",
    "phonemes_by_frames",
    "concatenate_features",
    "reverse",
    "random_chunk",
    "ContoursExtractor",
]

LOGGER = logging.getLogger("root")

REAL_NUMBER_REGEXP = re.compile(r"[0-9]*\.?[0-9]+")


def _fp_eq(a, b, eps=1e-5):
    return np.abs(np.float32(a) - np.float32(b)) < eps


def get_n_tokens(
    token,
    quantity: str,
    step: float,
    max_durations: float = 5.0,
    correction_factor: float = 1.0,
    breath_power: tp.Optional[str] = None,
    breath_dura: tp.Optional[str] = None,
    num_tokens_for_breath: int = 5,
) -> tp.List[Token]:
    try:
        seconds = 0.0
        if "ms" == quantity[-2:]:
            seconds = float(re.findall(REAL_NUMBER_REGEXP, quantity)[0]) / 1000
        elif "s" == quantity[-1]:
            seconds = float(re.findall(REAL_NUMBER_REGEXP, quantity)[0])
        seconds = max(0.05, min(max_durations, seconds * correction_factor))
        num_tokens = int(seconds / step)
    except Exception as e:  # TODO: more strict exception pool
        LOGGER.error(trace("get_n_tokens", e))
        return []

    token.phonemes = [TextProcessor.sil]
    pause_tokens = [copy(token) for _ in range(num_tokens)]

    if breath_power is not None:
        try:
            power = max(0.0, min(1.0, float(breath_power)))
            duration = num_tokens_for_breath
            if power > 0.5:
                duration = int(duration * 1.25)
            if breath_dura is not None:
                dura = max(0.0, min(2.0, float(breath_dura)))
                duration = int(duration * dura)

            offset = 1
            if duration > 0 and len(pause_tokens) >= duration + offset:
                win = signal.windows.tukey(duration, alpha=1.4).astype(np.float32) + 0.2
                for i in range(duration):
                    pause_tokens[-i - 1 - offset].meta["noise_level"] = [
                        power * 3.0 * win[i]
                    ]

        except Exception as e:
            LOGGER.error(trace("get_n_tokens", e))

    for token in pause_tokens:
        token.from_ssml = True

    return pause_tokens


@PipeRegistry.registry(inputs={"sent"}, outputs={"sent"})
def add_pauses_from_text(
    ds: TTSDataSample,
    level: str = "syntagmas",
    begin_pause: bool = True,
    end_pause: bool = True,
    num_symbols: int = 1,
    step: float = 0.05,
    max_sil_tokens: int = 10,
    pause_from_punct_map: tp.Optional[tp.Dict[str, str]] = None,
    pauses_with_punctuation: bool = False,
    pause_after_short_words: bool = False,
):
    sil_symbol = TextProcessor.sil
    normal_pause = Token(sil_symbol)
    weak_pause = Token(sil_symbol)
    strong_pause = Token(sil_symbol)

    breaks = {
        idx: insertions["break"]
        for idx, insertions in ds.sent.ssml_insertions
        if insertions.get("break") is not None
    }

    for break_params in breaks.values():
        if "breath" in break_params and break_params["breath"] in ["1", "true", "enable"]:
            if "breath_power" not in break_params:
                break_params["breath_power"] = 0.6

    global_idx = 0
    for synt in ds.sent.syntagmas:
        this_syntagma_tokens = []
        for token in synt.tokens:
            this_syntagma_tokens.append(token)
            if global_idx in breaks:
                this_syntagma_tokens.extend(
                    get_n_tokens(
                        copy(normal_pause),
                        breaks[global_idx]["time"],
                        step=step,
                        breath_power=breaks[global_idx].get("breath_power"),
                        breath_dura=breaks[global_idx].get("breath_dura"),
                    )
                )
            global_idx += 1
        synt.tokens = this_syntagma_tokens

    is_first_pause_from_ssml = breaks.get(-1) is not None
    if is_first_pause_from_ssml:
        ds.sent.syntagmas[0].tokens = (
            get_n_tokens(
                copy(normal_pause),
                breaks[-1]["time"],
                step=step,
                breath_power=breaks[-1].get("breath_power"),
                breath_dura=breaks[-1].get("breath_dura"),
            )
            + ds.sent.syntagmas[0].tokens
        )

    if ds.pauses_durations is not None:
        if level == "syntagmas":
            num_sil_tokens = len(ds.pauses_durations)
            pause_tokens = [normal_pause] * num_sil_tokens
            for i in range(len(pause_tokens)):
                num_sil_tokens = int(ds.pauses_durations[i].numpy()[0] // step)
                pause_tokens[i].phonemes = (sil_symbol,) * min(
                    num_sil_tokens, max_sil_tokens
                )
        else:
            raise ValueError("Words pauses level is not supported")
    else:
        if level == "syntagmas":
            normal_pause.phonemes = (sil_symbol,) * num_symbols
            weak_pause.phonemes = (sil_symbol,) * max(1, num_symbols // 2)
            strong_pause.phonemes = (sil_symbol,) * max(1, num_symbols * 2)
        else:
            normal_pause.phonemes = (sil_symbol,)
            weak_pause.phonemes = (sil_symbol,)
            strong_pause.phonemes = (sil_symbol,)

    i = 1 if begin_pause else 0
    for synt_idx, synt in enumerate(ds.sent.syntagmas):
        if level == "syntagmas":
            new_tokens = []
            for token_idx, token in enumerate(synt.tokens):
                if not token.is_punctuation:
                    new_tokens.append(token)
                    continue

                prev_token = next_token = None
                if token != synt.tokens[0]:
                    prev_token = synt.tokens[token_idx - 1]
                if token != synt.tokens[-1]:
                    next_token = synt.tokens[token_idx + 1]
                if prev_token is None:
                    if synt_idx > 0:
                        prev_token = ds.sent.syntagmas[synt_idx - 1].tokens[-1]
                if next_token is None:
                    if synt != ds.sent.syntagmas[-1]:
                        next_token = ds.sent.syntagmas[synt_idx + 1].tokens[0]

                is_break_pause = (
                    prev_token is not None
                    and prev_token.is_pause
                    and prev_token.from_ssml
                ) or (
                    next_token is not None
                    and next_token.is_pause
                    and next_token.from_ssml
                )

                if not is_break_pause and (
                    synt != ds.sent.syntagmas[-1] or token_idx == 0
                ):
                    symb = token.text
                    if ds.pauses_durations is not None:
                        new_tokens.append(copy(pause_tokens[i]))
                    elif pause_from_punct_map and symb and symb in pause_from_punct_map:
                        if pause_from_punct_map[symb] == "normal":
                            new_tokens.append(copy(normal_pause))
                        elif pause_from_punct_map[symb] == "weak":
                            new_tokens.append(copy(weak_pause))
                        elif pause_from_punct_map[symb] == "strong":
                            new_tokens.append(copy(strong_pause))
                    else:
                        new_tokens.append(copy(normal_pause))

                new_tokens.append(token)
                i += 1

            synt.tokens = new_tokens

        elif level == "words":
            group = TokenUtils.group_tokens_by_word(synt.tokens)

            def get_asr_pause_dura(tokens_: tp.List[Token]) -> float:
                try:
                    word = [t for t in tokens_ if t.is_word][0]
                    return 0.0 if word.asr_pause is None else float(word.asr_pause)
                except Exception:
                    return 0.0

            for i, tokens in enumerate(group[:-1]):
                asr_pause = get_asr_pause_dura(tokens)
                check_pos = any(
                    t.pos in ["ADP", "CCONJ", "SCONJ", "DET", "PART"] for t in tokens
                )
                if (
                    not pause_after_short_words
                    and check_pos
                    and asr_pause == 0.0
                    and not (tokens[-1].is_punctuation or group[i + 1][0].is_punctuation)
                ):
                    continue

                tokens.append(copy(normal_pause))

            if synt != ds.sent.syntagmas[-1]:
                pause = copy(normal_pause)
                last_token = synt.tokens[-1]
                if pauses_with_punctuation and last_token.is_punctuation:
                    pause.phonemes = (f"<{last_token.text}>{pause.text}",)

                group[-1].append(pause)

            synt.tokens = list(itertools.chain.from_iterable(group))

        else:
            raise NotImplementedError

    if begin_pause:
        if ds.pauses_durations is not None:
            weak_pause = pause_tokens[0]
        if not is_first_pause_from_ssml:
            ds.sent.syntagmas[0].tokens.insert(0, copy(weak_pause))

    if end_pause:
        last_syntagma_tokens_reversed = [
            x for x in ds.sent.syntagmas[-1].tokens[::-1] if not x.is_punctuation
        ]
        is_last_pause_from_ssml = last_syntagma_tokens_reversed[0].text == sil_symbol
        if ds.pauses_durations is not None:
            strong_pause = pause_tokens[i]
        if not is_last_pause_from_ssml:
            if ds.sent.syntagmas[-1].tokens[-1].text != sil_symbol:
                pause = copy(strong_pause)
                last_token = ds.sent.syntagmas[-1].tokens[-1]
                if pauses_with_punctuation and last_token.is_punctuation:
                    pause.phonemes = (f"<{last_token.text}>{pause.text}",)

                ds.sent.syntagmas[-1].tokens.append(pause)

    all_tokens = [synt.tokens for synt in ds.sent.syntagmas]
    all_tokens = list(itertools.chain.from_iterable(all_tokens))
    ds.sent.tokens = all_tokens
    return ds


@PipeRegistry.registry(
    inputs={"sent", "word_timestamps", "phoneme_timestamps"},
    outputs={"sent", "word_timestamps", "phoneme_timestamps"},
)
def add_pauses_from_timestamps(
    ds: TTSDataSample,
    min_len: float = 0.001,  # in seconds
    step: tp.Optional[float] = None,  # in seconds
    calc_noise_level: bool = False,
    use_pauses_from_asr: bool = False,
    check_phoneme_length: bool = False,
):
    sil_symbol = TextProcessor.sil

    # add dummy EOS token
    ds.sent.syntagmas[-1].tokens.append(Token(TextProcessor.eos))
    ds.word_timestamps.append([(ds.audio_chunk.duration - 9e-6, ds.audio_chunk.duration)])
    ds.phoneme_timestamps.append(
        Timestamps(
            np.asarray([(ds.audio_chunk.duration - 9e-6, ds.audio_chunk.duration)])
        )
    )

    begin_token = ds.sent.tokens[0]
    end_token = ds.sent.syntagmas[-1].tokens[-1]

    prev_ts = 0
    word_idx = 0
    ts_words_processed: tp.List = []
    ts_phonemes_processed: tp.List[Timestamps] = []
    for synt in ds.sent.syntagmas:
        tokens_processed = []
        for token in synt.tokens:
            if token.is_punctuation:
                tokens_processed.append(token)
                continue
            ts_word = ds.word_timestamps[word_idx]
            ts_ph = ds.phoneme_timestamps[word_idx]

            is_anomaly_phoneme = False
            if check_phoneme_length:
                longer_phoneme_l = np.diff(ts_ph.intervals).max()
                if longer_phoneme_l > 0.2:
                    is_anomaly_phoneme = True

                if word_idx != len(ds.phoneme_timestamps) - 1:
                    ts_ph_r = ds.phoneme_timestamps[word_idx + 1]
                    longer_phoneme_r = np.diff(ts_ph_r.intervals).max()
                    if longer_phoneme_r > 0.2:
                        is_anomaly_phoneme = True

            asr_pause = 0
            if use_pauses_from_asr:
                asr_pause = float(token.asr_pause) if token.asr_pause is not None else 0

            if not _fp_eq(ts_word[0], prev_ts):
                diff = ts_word[0] - prev_ts
                if (
                    diff > min_len
                    or token in [begin_token, end_token]
                    or asr_pause > 0.1
                    or is_anomaly_phoneme
                ):
                    if step is None:
                        num_sil_tokens = 1
                    else:
                        num_sil_tokens = max(math.floor(diff / step), 1)

                    pause_token = Token(sil_symbol)
                    pause_token.phonemes = (sil_symbol,) * num_sil_tokens

                    new_ts_word = [prev_ts, ts_word[0]]

                    seq_idx = np.arange(0, num_sil_tokens)
                    sil_ts = np.dstack((seq_idx, seq_idx + 1)).astype(np.float32)[0]
                    new_ts_ph = prev_ts + sil_ts * (1 if step is None else step)
                    new_ts_ph[-1][1] = ts_word[0]

                    tokens_processed.append(pause_token)
                    ts_words_processed.append(new_ts_word)
                    ts_phonemes_processed.append(Timestamps(new_ts_ph))

                    if calc_noise_level:
                        noise_level = pause_token.meta.setdefault("noise_level", [])
                        for (a, b) in new_ts_ph:
                            a = int(a * ds.audio_chunk.sr)
                            b = int(b * ds.audio_chunk.sr)
                            noise_level.append(detect_noise(ds.audio_chunk.waveform[a:b]))
                else:
                    if token.text == TextProcessor.eos:
                        ts_words_processed[-1][1] += diff
                        ts_phonemes_processed[-1][-1][1] += diff
                    elif ts_words_processed:
                        ts_word[0] -= diff / 2
                        ts_ph[0][0] -= diff / 2
                        ts_words_processed[-1][1] += diff / 2
                        ts_phonemes_processed[-1][-1][1] += diff / 2
                    else:
                        ts_word[0] -= diff
                        ts_ph[0][0] -= diff

            assert _fp_eq(ts_word[0], ts_ph[0][0]) and _fp_eq(ts_word[1], ts_ph[-1][1])
            if ts_words_processed:
                assert _fp_eq(ts_words_processed[-1][1], ts_word[0])

            if token.text == TextProcessor.eos:
                break

            word_idx += 1
            prev_ts = ts_word[1]
            tokens_processed.append(token)
            ts_words_processed.append(ts_word)
            ts_phonemes_processed.append(ts_ph)

        synt.tokens = tokens_processed

    all_tokens = [synt.tokens for synt in ds.sent.syntagmas]
    ds.sent.tokens = list(itertools.chain.from_iterable(all_tokens))
    ds.word_timestamps = Timestamps(np.asarray(ts_words_processed))
    ds.phoneme_timestamps = ts_phonemes_processed

    assert _fp_eq(ds.word_timestamps.duration, ds.audio_chunk.duration)
    return ds


@PipeRegistry.registry(
    inputs={"word_timestamps", "phoneme_timestamps", "magnitude", "hop_len"},
    outputs={"durations"},
)
def calc_durations(
    ds: TTSDataSample,
    as_int: bool = False,
    in_seconds: bool = False,
    token_level: bool = False,
):
    hop_len = ds.get_param_val("hop_len")
    ds.aggregated = {} if ds.aggregated is None else ds.aggregated

    ts_phonemes = Timestamps(np.concatenate(ds.phoneme_timestamps))
    if in_seconds:
        if token_level:
            ts_words = ds.word_timestamps
            assert _fp_eq(ts_words.begin, ts_phonemes.begin) and _fp_eq(
                ts_words.end, ts_phonemes.end
            )
            word_durations = np.diff(ts_words).squeeze(1)
        durations = np.diff(ts_phonemes).squeeze(1)
    else:
        spec_len = ds.magnitude.shape[0]

        if token_level:
            ts_words = ds.word_timestamps
            ts_words = ts_words.to_frames(
                hop_len / ds.audio_chunk.sr, ds.magnitude.shape[0], as_int=as_int
            )
            word_durations = np.diff(ts_words).squeeze(1)

        ts_phonemes = ts_phonemes.to_frames(
            hop_len / ds.audio_chunk.sr, ds.magnitude.shape[0], as_int=as_int
        )
        durations = np.diff(ts_phonemes).squeeze(1)

        if as_int:
            durations = durations.astype(np.int64)
            if token_level:
                word_durations = word_durations.astype(np.int64)
        else:
            durations *= spec_len / durations.sum()
            if token_level:
                word_durations *= spec_len / word_durations.sum()

        total_duration = durations.sum().astype(np.int64)
        if spec_len - total_duration == 1:
            # TODO: find where melshape becomes +1 frame
            durations[-1] += 1
            if token_level:
                if spec_len - word_durations.sum().astype(np.int64):
                    word_durations[-1] += 1
        elif spec_len > total_duration:
            raise ValueError(f"Size mismatch between mel and durations at {ds.file_path}")

    if ds.symbols:
        if durations.shape[0] != len(ds.symbols):
            raise ValueError(
                f"Number of instances mismatch between symbols and durations at {ds.file_path}"
            )
    # elif token_level:
    #     if durations.shape[0] != len(ds.ling_feat["pos_tags"]):
    #         raise ValueError(
    #             "Number of instances mismatch between linguistic features and durations at {ds.file_path}."
    #         )

    if not token_level:
        t_durations = []
        p = 0
        for t in ds.sent.tokens:
            if t.is_punctuation:
                continue

            a, b = p, p + t.num_phonemes
            t_durations.append(durations[a:b].sum())
            p = b

        t_durations = np.array(t_durations)
        assert _fp_eq(t_durations.sum(), durations.sum(), 1e-1)

        if as_int:
            t_durations = t_durations.astype(np.int64)

        ds.aggregated["word_durations"] = t_durations
    else:
        ds.aggregated["word_durations"] = word_durations

    ds.durations = durations
    return ds


@PipeRegistry.registry(
    inputs={"durations", "magnitude"},
    outputs={"invert_durations"},
)
def calc_invert_durations(ds: TTSDataSample, token_level: bool = False):
    invert = []
    for dura in ds.durations:
        invert += [1 / dura] * dura

    ds.invert_durations = np.array(invert, dtype=np.float32)

    if token_level:
        invert = []
        for d in ds.aggregated["word_durations"]:
            invert += [1 / d] * d

        ds.aggregated["word_invert_durations"] = np.array(invert, dtype=np.float32)

    return ds


@PipeRegistry.registry(inputs={"durations"}, outputs={"aggregate"})
def aggregate_by_phoneme(
    ds: TTSDataSample,
    attributes: tp.Union[tp.List[str], str],
    agg: str = "mean",
):
    """Apply aggregation function to TTSDataSample attribute in time domain.

    Parameters
    ----------
    ds: DataSample
    attributes : string or sequence of strings.
    agg : string or callable.

    """
    if agg == "mean":
        agg_func = np.mean
    elif agg == "median":
        agg_func = np.median  # type: ignore
    elif agg == "custom":

        def custom_agg(x, axis=0, *args, **kwargs):
            return np.array(
                [np.mean(x, axis=axis), np.max(x, axis=axis), np.min(x, axis=axis)]
            ).reshape(1, -1)

        agg_func = custom_agg  # type: ignore
    elif agg == "range_diff":

        def range_diff_agg(x, axis=0, *args, **kwargs):
            if x.shape[axis] > 2:
                dx = np.diff(x, n=1)
            else:
                dx = [0.0, 0.0]
            return np.array(
                [
                    np.mean(x, axis=axis),
                    np.mean(dx, axis=axis),
                    np.max(x, axis=axis) - np.min(x, axis=axis),
                ]
            ).reshape(1, -1)

        agg_func = range_diff_agg  # type: ignore
    elif agg == "diff":

        def diff_agg(x, axis=0, *args, **kwargs):
            if x.shape[axis] > 3:
                dx = np.diff(x, n=1)
                d2x = np.diff(x, n=2)
            else:
                dx = [0.0, 0.0]
                d2x = [0.0, 0.0]
            return np.array(
                [np.mean(x, axis=axis), np.mean(dx, axis=axis), np.mean(d2x, axis=axis)]
            ).reshape(1, -1)

        agg_func = diff_agg  # type: ignore
    else:
        raise NotImplementedError

    attributes = [attributes] if isinstance(attributes, str) else attributes
    attributes_data = {attr: getattr(ds, attr, None) for attr in attributes}
    result: tp.Dict = {attr: list() for attr in attributes}

    frame_ts = np.concatenate([[0], np.cumsum(ds.durations)]).astype(np.int64)
    for start, end in zip(frame_ts[0:-1], frame_ts[1:]):
        for attr in attributes:
            data = attributes_data[attr]
            if data is None:
                raise KeyError(f"Attribute '{attr}' not found in TTSDataSample.")

            if data.ndim == 2:
                feat_size = data.shape[1]
            else:
                feat_size = 1

            if end - start >= 1:
                phoneme_attr = agg_func(data[start:end], axis=0)
            else:
                if start < len(data):
                    if agg == "custom":
                        phoneme_attr = np.repeat(data[start], 3).reshape(1, feat_size * 3)
                    elif agg == "diff":
                        phoneme_attr = np.array([data[start], 0.0, 0.0]).reshape(
                            1, feat_size * 3
                        )
                    elif agg == "range_diff":
                        phoneme_attr = np.array([data[start], 0.0, 0.0]).reshape(
                            1, feat_size * 3
                        )
                    else:
                        phoneme_attr = data[start]
                else:
                    phoneme_attr = (
                        0.0 if data.ndim == 1 else np.zeros((feat_size), dtype=data.dtype)
                    )
            result[attr].append(phoneme_attr)

    if ds.aggregated is None:
        ds.aggregated = {}

    for attr in attributes:
        data = np.stack(result[attr]).astype(np.float32)
        data = data.squeeze()
        assert (
            data.shape[0] == ds.durations.shape[0]
        ), f"Shapes mismatch after aggr {data.shape[0], ds.durations.shape[0]}"
        ds.aggregated[attr] = data

    return ds


@PipeRegistry.registry(inputs={"durations"}, outputs={"aggregate"})
def curvature_estimate_by_phoneme(
    ds: TTSDataSample,
    attributes: tp.Union[tp.List[str], str],
):
    """Apply curvature estimate function to TTSDataSample attribute in time domain.

    Parameters
    ----------
    ds: DataSample
    attributes : string or sequence of strings.

    """

    def calc_angle(p1, p2, p3):
        angle = math.atan2(p3["y"] - p1["y"], p3["x"] - p1["x"]) - math.atan2(
            p2["y"] - p1["y"], p2["x"] - p1["x"]
        )
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle

    # def med(x, y, z):
    #    return math.sqrt(abs(2 * (x * x + z * z) - y * y)) / 2

    # def dist(p1, p2):
    #    return math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])

    def curv_func(x):
        if len(x) == 1 or np.sum(x) < 0.5:
            return np.zeros(2)

        p1 = {"x": 0, "y": x[0]}
        p1x = {"x": 1, "y": x[0]}
        p3 = {"x": len(x), "y": x[-1]}
        p3x = {"x": len(x) - 1, "y": x[-1]}

        if len(x) == 2:
            p2 = p3
            alpha_1 = calc_angle(p1, p2, p1x)
            alpha_2 = alpha_1
        else:
            if len(x) == 3:
                p2 = {"x": 1, "y": x[1]}
            else:
                y = x[1:-1]
                p2 = {"x": np.argsort(y)[len(y) // 2] + 1, "y": np.median(y)}

            alpha_1 = calc_angle(p1, p2, p1x)
            alpha_2 = calc_angle(p3, p2, p3x)

        # ax.plot([p1["x"] + start, p2["x"] + start], [80 - p1["y"] / 10, 80 - p2["y"] / 10])
        # ax.plot([p3["x"] + start, p2["x"] + start], [80 - p3["y"] / 10, 80 - p2["y"] / 10])
        # ax.text(p1["x"] + start, 80 - p1["y"] / 10, str(round(alpha_1,3)))
        # ax.text(p3["x"] + start, 80 - p3["y"] / 10, str(round(alpha_2,3)),color='red')
        return np.array([alpha_1, alpha_2])

    attributes = [attributes] if isinstance(attributes, str) else attributes
    attributes_data = {attr: getattr(ds, attr, None) for attr in attributes}
    result: tp.Dict = {attr: list() for attr in attributes}

    frame_ts = np.concatenate([[0], np.cumsum(ds.durations)]).astype(np.int64)
    for start, end in zip(frame_ts[0:-1], frame_ts[1:]):
        for attr in attributes:
            data = attributes_data[attr]
            if data is None:
                raise KeyError(f"Attribute '{attr}' not found in TTSDataSample.")

            assert data.ndim == 1
            if end - start == 1:
                a = start
                b = end + 1
            else:
                a = start
                b = end
            result[attr].append(curv_func(data[a:b]))

    if ds.aggregated is None:
        ds.aggregated = {}

    for attr in attributes:
        data = np.stack(result[attr]).astype(np.float32)
        data = data.squeeze()
        assert (
            data.shape[0] == ds.durations.shape[0]
        ), f"Shapes mismatch after aggr {data.shape[0], ds.durations.shape[0]}"
        ds.aggregated[f"curv_{attr}"] = data

    return ds


@PipeRegistry.registry(inputs={"magnitude"}, outputs={"gate"})
def add_gate_value(ds: TTSDataSample):
    ds.gate = np.zeros((ds.magnitude.shape[0],), dtype=np.float32)
    ds.gate[-1] = 1.0
    return ds


@PipeRegistry.registry(
    inputs={"sent", "hop_len"},
    outputs={"sent"},
    optional={"audio_chunk", "word_timestamps", "phoneme_timestamps"},
)
def add_service_tokens(ds: TTSDataSample):
    bos_token = Token(TextProcessor.bos)
    bos_token.phonemes = (TextProcessor.bos,)

    eos_token = Token(TextProcessor.eos)
    eos_token.phonemes = (TextProcessor.eos,)

    synt_begin = ds.sent.syntagmas[0]
    synt_end = ds.sent.syntagmas[-1]

    synt_begin.tokens.insert(0, bos_token)
    synt_end.tokens.append(eos_token)

    hop_len = ds.get_param_val("hop_len")

    if ds.word_timestamps and ds.phoneme_timestamps:
        frame_dura = 0.25 * hop_len / ds.audio_chunk.sr
        ts_end = ds.word_timestamps.end

        ds.word_timestamps.append_left([(0.0, frame_dura)])
        ds.word_timestamps.append([(ts_end - frame_dura, ts_end)])

        ds.phoneme_timestamps[0].append_left([(0.0, frame_dura)])
        ds.phoneme_timestamps.insert(
            0, Timestamps(np.asarray([ds.phoneme_timestamps[0][0]]))
        )
        ds.phoneme_timestamps[1].delete(0)

        ds.phoneme_timestamps[-1].append([(ts_end - frame_dura, ts_end)])
        ds.phoneme_timestamps.append(
            Timestamps(np.asarray([ds.phoneme_timestamps[-1][-1]]))
        )
        ds.phoneme_timestamps[-2].delete(-1)

    all_tokens = [synt.tokens for synt in ds.sent.syntagmas]
    ds.sent.tokens = list(itertools.chain.from_iterable(all_tokens))
    return ds


@PipeRegistry.registry(
    inputs={"sent", "transcription", "durations"},
    outputs={"transcription"},
)
def phonemes_by_frames(ds: TTSDataSample):
    transcription = ds.transcription
    durations = ds.durations

    ext_transcription = []
    for d, t in zip(durations, transcription):
        ext_transcription += [t] * d

    ds.transcription_by_frames = np.array(ext_transcription)
    assert ds.magnitude.shape[0] == ds.transcription_by_frames.shape[0]
    return ds


@PipeRegistry.registry(inputs={"energy", "pitch", "durations"}, outputs={"concatenate"})
def concatenate_features(
    ds: TTSDataSample,
    attributes: tp.Union[tp.List[str], str],
):
    feats = []
    for attr in attributes:
        feat = getattr(ds, attr)
        if isinstance(feat, tp.MutableMapping):
            feat = list(feat.values())
        elif not isinstance(feat, tp.MutableSequence):
            feat = [feat]

        for item in feat:
            if item.ndim == 1:
                item = np.expand_dims(item, axis=1)
            else:
                item = item

            assert item.ndim == 2, ValueError("feature must have two dimensions!")
            feats.append(item.astype(np.float32))

    ds.concatenate = np.concatenate(feats, axis=1)
    return ds


@PipeRegistry.registry(inputs={"mel", "transcription"}, outputs={"mel", "transcription"})
def reverse(ds: TTSDataSample, p: float = 0.15):
    if random.random() > p:
        return ds

    ds.magnitude = np.flipud(ds.magnitude).copy()
    ds.mel = np.flipud(ds.mel).copy()

    ds.transcription = np.flipud(ds.transcription).copy()

    for k, v in ds.ling_feat.items():
        ds.ling_feat[k] = np.flipud(v).copy()

    if ds.ssl_feat is not None:
        ds.ssl_feat.encode = np.flipud(ds.ssl_feat.encode).copy()

    if ds.symbols is not None:
        ds.symbols = ds.symbols[::-1]

    if ds.synt_lengths is not None:
        ds.synt_lengths = np.flipud(ds.synt_lengths).copy()

    if ds.word_lengths is not None:
        ds.word_lengths = np.flipud(ds.word_lengths).copy()

    if ds.lm_feat is not None:
        ds.lm_feat = np.flipud(ds.lm_feat).copy()

    return ds


@PipeRegistry.registry(
    inputs={"audio_chunk", "sent", "word_timestamps"},
    outputs={"audio_chunk", "sent"},
)
def random_chunk(ds: TTSDataSample, min_length: float = 2.0, max_length: float = 4.0):
    sr = ds.audio_chunk.sr
    chunk_dura = ds.audio_chunk.duration
    hop_len = ds.get_param_val("hop_len")

    if chunk_dura > max_length:
        length = min_length + random.random() * (max_length - min_length)
        a = random.random() * (chunk_dura - length)
        b = a + length

        ts_by_word = ds.word_timestamps
        begin_idx, begin_ts = 0, chunk_dura
        end_idx, end_ts = len(ts_by_word) - 1, chunk_dura

        for i, (t0, t1) in enumerate(ts_by_word):
            if abs(a - t0) < abs(a - begin_ts) and a < t0:  # type: ignore
                begin_idx = i
                begin_ts = t0  # type: ignore
            if abs(b - t1) < abs(b - end_ts) and b > t1:  # type: ignore
                end_idx = i
                end_ts = t1  # type: ignore

        if ds.audio_chunk.is_trim:
            ds.audio_chunk.data = ds.audio_chunk.data[
                int(begin_ts * sr) : int(end_ts * sr)
            ]
            ds.audio_chunk.end = end_ts - begin_ts
        else:
            ds.audio_chunk.begin = begin_ts
            ds.audio_chunk.end = end_ts

        word_tokens = ds.sent.get_words()
        word_tokens = word_tokens[begin_idx : end_idx + 1]
        begin_token = ds.sent.tokens.index(word_tokens[0])
        end_token = ds.sent.tokens.index(word_tokens[-1])

        ds.sent.tokens = ds.sent.tokens[begin_token : end_token + 1]
        ds.sent.syntagmas = [Syntagma(ds.sent.tokens)]
        ds.word_timestamps = Timestamps(ds.word_timestamps[begin_idx : end_idx + 1])
        ds.phoneme_timestamps = ds.phoneme_timestamps[begin_idx : end_idx + 1]

        ds.word_timestamps -= begin_ts
        ds.phoneme_timestamps = [ts - begin_ts for ts in ds.phoneme_timestamps]

        if ds.mu_law_waveform is not None:
            ds.mu_law_waveform = ds.mu_law_waveform[int(begin_ts * sr) : int(end_ts * sr)]

        if ds.transcription is not None:
            if ds.transcription.shape[0] == ds.magnitude.shape[0]:
                a = int(begin_ts * sr / hop_len)
                b = int(end_ts * sr / hop_len)
                ds.transcription = ds.transcription[a : b + 1]
            else:
                ds.transcription = ds.transcription[begin_token : end_token + 1]

    if ds.audio_chunk.duration > max_length:
        raise ValueError(f"audio_chunk: {ds.audio_chunk.duration} ")

    return ds


class ContoursExtractor:
    def __init__(
        self,
        index_filename: str,
        labels_filename: bool = False,
        n: int = 100,
    ):
        self.t = AnnoyIndex(n - 20, "euclidean")
        self.t.load(index_filename)
        self.labels = np.load(labels_filename)
        self.n = n

    @PipeRegistry.registry(
        inputs={"durations", "sent", "word_lengths", "pitch"}, outputs={"aggregate"}
    )
    def process(
        self,
        ds: TTSDataSample,
    ):
        """Apply curvature estimate function to TTSDataSample attribute in time domain.

        Parameters
        ----------
        ds: DataSample

        """

        words_length = ds.word_lengths
        frame_ts_word = np.concatenate([[0], np.cumsum(ds.word_lengths)]).astype(np.int64)
        frame_ts = np.around(np.concatenate([[0], np.cumsum(ds.durations)])).astype(
            np.int64
        )
        pitch = getattr(ds, "pitch", None)
        tokens = [token.text for token in ds.sent.tokens if token.pos != "PUNCT"]

        indices = []

        for idx, (start, end) in enumerate(zip(frame_ts_word[0:-1], frame_ts_word[1:])):
            if idx < len(tokens):
                if tokens[idx] not in ["<BOS>", "<EOS>", "<SIL>"]:
                    frame = frame_ts[start : end + 1]
                    first_ind = frame[0]
                    last_ind = min(frame[-1], pitch.shape[0] - 1)
                    contour = pitch[first_ind:last_ind]
                    if contour.shape[0] == 1:
                        contour = np.concatenate((contour, contour))
                    x = np.arange(0, contour.shape[0])
                    f = interpolate.interp1d(x, contour, fill_value="extrapolate")
                    xnew = np.arange(0, contour.shape[0], contour.shape[0] / self.n)
                    contour = f(xnew)[: self.n][10:-10]
                    contour = contour - contour.mean()
                    indices.extend(
                        [self.labels[self.t.get_nns_by_vector(contour, 1)][0]]
                        * words_length[idx]
                    )
                else:
                    indices.extend([-1] * words_length[idx])

        indices = np.array(indices)
        assert (
            indices.shape[0] == ds.durations.shape[0]
        ), f"Shapes mismatch after aggr {indices.shape[0], ds.durations.shape[0]}"
        ds.aggregated["pitch_contour"] = indices

        return ds


if __name__ == "__main__":
    _lang = "KK"

    text_parser = TextParser(lang=_lang)
    utterance = """

        <break time="0.3s"/> в <audio> <break time="0.1s"/> том самом, <break time="0.200s"/>
        голодном <break time="0.1s"/> девяностом году <break time="0.28s"/> был <break time="200ms"/>
        открыт этот музей <break time="200ms"/></audio>.

    """
    doc = text_parser.process(Doc(utterance))
    sent = doc.sents[0]
    _ds = TTSDataSample(sent=sent)
    _ds = add_pauses_from_text(_ds)
    print(_ds.sent.text)

    text_processor = TextProcessor(lang=_lang)
    _ds = text_processor.process(_ds)
    print(_ds.transcription)
    print("num_symbols_per_phoneme_token:", text_processor.num_symbols_per_phoneme_token)
