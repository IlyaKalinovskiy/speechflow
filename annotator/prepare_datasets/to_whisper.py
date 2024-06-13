import argparse

from pathlib import Path

from speechflow.data_pipeline.datasample_processors.audio_processors import (
    SignalProcessor,
)
from speechflow.data_pipeline.datasample_processors.data_types import AudioDataSample
from speechflow.data_pipeline.dataset_parsers import EasyDSParser


def _to_whisper(wav_path: Path):
    if wav_path.with_suffix(".whisp.wav").exists():
        return True

    ds = AudioDataSample(file_path=wav_path)

    signal_proc = SignalProcessor(("load", "whisper"))
    ds = signal_proc.process(ds)

    ds.audio_chunk.save(wav_path.with_suffix(".whisp.wav"))
    return True


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="Convert to whisper dataset")
    arguments_parser.add_argument(
        "-d", "--data_root", help="path to dataset", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    args = arguments_parser.parse_args()

    data_root = args.data_root

    parser = EasyDSParser(func=_to_whisper)
    data = parser.run_in_dir(
        data_root=data_root,
        file_extension=".wav",
        n_processes=args.n_processes,
    )

    print(f"DONE! Prepare {len(data)} files")
