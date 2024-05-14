import json

from itertools import groupby
from pathlib import Path

from multilingual_text_parser import Token

from annotator.asr_services import OpenAIASR
from annotator.audiobook_spliter import AudiobookSpliter
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.data_pipeline.datasample_processors.tts_processors import (
    TextProcessor,
    add_pauses_from_text,
)


def forced_alignment(
    text: str, wav_path: Path, asr: OpenAIASR, spliter: AudiobookSpliter
):
    transc_path = wav_path.with_suffix(".whisper")

    if not transc_path.exists():
        data = asr.read_datasamples([wav_path], n_processes=1)
        transcription = data.item(0)
    else:
        json_dump = transc_path.read_text(encoding="utf-8")
        transcription = json.loads(json_dump)

    metadata = {
        "wav_path": wav_path,
        "text": text,
        "transcription": transcription,
    }
    metadata = spliter.do_preprocessing([metadata], spliter.preproc_fn)
    metadata = spliter.to_datasample(metadata)

    results = []
    for sega in metadata.item(0)["segmentation"]:
        results.append({"text": sega.sent.text_orig})

        new_tokens = []
        for token in sega.sent.tokens:
            new_tokens.append(token)
            try:
                if token.asr_pause and float(token.asr_pause) > 0:
                    pause = Token(TextProcessor.sil)
                    pause.meta["duration"] = float(token.asr_pause)
                    new_tokens.append(pause)
            except:
                pass
        results[-1]["text_with_pauses_from_asr"] = new_tokens

        new_tokens = []
        for token in sega.auxiliary["transcription"]:
            prev_word_ts = token.meta.get("prev_word_ts")
            next_word_ts = token.meta.get("next_word_ts")

            if prev_word_ts is None and token.meta["ts"][0] > 0:
                pause = Token(TextProcessor.sil)
                pause.meta["duration"] = token.meta["ts"][0]
                new_tokens.append(pause)

            new_tokens.append(token)

            if next_word_ts is not None and next_word_ts[0] > token.meta["ts"][1]:
                pause = Token(TextProcessor.sil)
                pause.meta["duration"] = next_word_ts[0] - token.meta["ts"][1]
                new_tokens.append(pause)

            if next_word_ts is None and token.meta["eos_ts"] > token.meta["ts"][1]:
                pause = Token(TextProcessor.sil)
                pause.meta["duration"] = token.meta["eos_ts"] - token.meta["ts"][1]
                new_tokens.append(pause)

        results[-1]["transcription_with_pauses"] = new_tokens

        tts_ds = TTSDataSample(sent=sega.sent)
        tts_ds = add_pauses_from_text(
            tts_ds,
            num_symbols=2,
            pause_from_punct_map={
                ",": "normal",
                "-": "weak",
                "—": "normal",
                ".": "strong",
            },
        )
        new_tokens = []
        for key, group_items in groupby(tts_ds.sent.tokens, key=lambda x: x.is_pause):
            if key:
                pause = Token(TextProcessor.sil)
                pause.meta["duration"] = 0.05 * sum([x.num_phonemes for x in group_items])
                new_tokens.append(pause)
            else:
                new_tokens += group_items

        results[-1]["text_with_pauses_from_punctuation"] = new_tokens

    return results


if __name__ == "__main__":
    asr = OpenAIASR(lang="RU")
    spliter = AudiobookSpliter(lang="RU")

    text_path = Path("P:/asr/def_tts_RU_Elena.txt")
    wav_path = Path("P:/asr/def_tts_RU_Elena.wav")

    text = text_path.read_text(encoding="utf-8")
    results = forced_alignment(text, wav_path, asr, spliter)

    for item in results:
        print(item["text"])
