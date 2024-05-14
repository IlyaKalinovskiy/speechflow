from pathlib import Path
from typing import List

from speechflow.data_pipeline.core import PipelineComponents
from speechflow.data_pipeline.datasample_processors.data_types import TTSDataSample
from speechflow.io import AudioSeg, Config, construct_file_list
from speechflow.utils.fs import get_root_dir


def main():
    config_data_file = (
        get_root_dir() / "tts/acoustic_models/configs/tts/tts_data_24KHz.yml"
    )
    cfg_data = Config.create_from_file(config_data_file)
    pipeline = PipelineComponents(cfg_data, "train")

    file_list = construct_file_list(
        data_root=get_root_dir() / "examples/simple_datasets/speech/SEGS",
        with_subfolders=True,
        ext=".TextGridStage3",
    )

    for file_path in file_list[:10]:
        sega = AudioSeg.load(file_path)
        metadata = {"file_path": file_path, "sega": sega}

        try:
            batch = pipeline.metadata_to_batch([metadata])
        except Exception as e:
            print(e)
            continue

        output_path = Path("temp")
        output_path.mkdir(exist_ok=True)

        samples: List[TTSDataSample] = batch.data_samples
        for idx, ds in enumerate(samples):
            sega = AudioSeg(ds.audio_chunk.trim(), ds.sent)
            sega.set_phoneme_timestamps(ds.phoneme_timestamps)

            fpath = output_path / f"{ds.audio_chunk.file_path.name}_{idx}.TextGridStage3"
            sega.save(fpath, with_audio=True)


if __name__ == "__main__":
    main()
