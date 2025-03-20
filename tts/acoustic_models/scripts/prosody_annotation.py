import typing as tp
import logging
import argparse

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from speechflow.data_pipeline.core import BaseBatchProcessor
from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.data_server.helpers import dataset_iterator
from speechflow.io import AudioSeg, Config
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from tts import acoustic_models
from tts.acoustic_models.scripts.dump import main as dump_worker

LOGGER = logging.getLogger("root")


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        help="path to model checkpoint",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-d",
        "--data_root",
        help="path to dataset",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-dump",
        "--dump_path",
        help="path to data dump",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", type=str, nargs="+"
    )
    arguments_parser.add_argument(
        "-bs",
        "--batch_size",
        help="batch size",
        type=int,
        default=16,
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    arguments_parser.add_argument(
        "-ngpu",
        "--n_gpus",
        help="number of GPU device for data_server",
        type=int,
        default=0,
    )
    arguments_parser.add_argument(
        "-md",
        "--model_device",
        help="device to process model",
        type=str,
        default="cpu",
    )
    arguments_parser.add_argument(
        "-n", "--n_components", help="number of prosodic classes", type=int, default=10
    )
    arguments_parser.add_argument(
        "-m", "--mapping_file", help="file to store mapping", type=Path, default=None
    )
    arguments_parser.add_argument(
        "--textgrid_ext_old",
        help="old extension of textgrid files",
        type=str,
        default=None,
    )
    arguments_parser.add_argument(
        "--textgrid_ext_new",
        help="new extension of textgrid files",
        type=str,
        default=".TextGrid_prosody",
    )
    return arguments_parser.parse_args()


def redefine_dataset(
    model,
    batch_processor: BaseBatchProcessor,
    labels: np.array,
    config: tp.Optional[Config] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    batch_size: int = 16,
    n_processes: int = 1,
    textgrid_ext_new: str = ".TextGrid_prosody",
):
    """Function that adds additional layer with prosody to segs."""

    iterator = dataset_iterator(
        config=config,
        value_select=value_select,
        batch_size=batch_size,
        n_processes=n_processes,
        device=str(batch_processor.device),
    )

    number_of_classes = defaultdict(int)
    for batch in tqdm(iterator, total=len(iterator)):
        model_inputs, _, _ = batch_processor(batch)

        with torch.no_grad():
            model_outputs = model(model_inputs)

        batch_indices = model_outputs.additional_content["encoding_indices_vq0_encoder_0"]

        for idx, ds in enumerate(batch.data_samples):
            tg_path = ds.file_path
            # if tg_path.with_suffix(textgrid_ext_new).exists():
            #    continue

            sega = AudioSeg.load(tg_path)
            if sega is None:
                continue

            tokens = [token.text for token in ds.sent.tokens if token.pos != "PUNCT"]
            indices = batch_indices[idx][: model_inputs.num_words[idx]]
            assert len(tokens) == indices.shape[0]

            indices = [
                labels[int(i.item())]
                for i, t in zip(indices, tokens)
                if t
                not in [
                    TTSTextProcessor.bos,
                    TTSTextProcessor.eos,
                    TTSTextProcessor.sil,
                ]
            ]

            j = 0
            for i, token in enumerate(sega.sent.tokens):
                if token.pos != "PUNCT":
                    ind = str(indices[j])
                    sega.sent.tokens[i].prosody = ind
                    j += 1
                    number_of_classes[ind] += 1
                else:
                    sega.sent.tokens[i].prosody = "-1"

            sega.meta["source_sega"] = tg_path.name
            sega.save(tg_path.with_suffix(textgrid_ext_new))

    total_number_of_words = sum(number_of_classes.values())
    print("Percent of each class and its absolute number:")
    print(
        "\n".join(
            [
                f"{cl}: {round(percentage/total_number_of_words*100, 2)}% ({percentage})"
                for cl, percentage in sorted(number_of_classes.items())
            ]
        )
    )


def main(
    ckpt_path: Path,
    data_root: tp.Optional[Path] = None,
    dump_path: tp.Optional[Path] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    batch_size: int = 16,
    n_processes: int = 1,
    n_gpus: int = 0,
    model_device: str = "cpu",
    n_components: int = 10,
    mapping_file: tp.Optional[Path] = None,
    textgrid_ext_old: tp.Optional[str] = None,
    textgrid_ext_new: str = ".TextGrid_prosody",
):
    checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
    cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

    if model_device == "cuda":
        model_device = torch.device(f"cuda:{get_freer_gpu()}")

    model_cls = getattr(acoustic_models, cfg_model["model"]["type"])
    model = model_cls(checkpoint["params"])
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(model_device)

    batch_processor_cls = getattr(acoustic_models, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()
    batch_processor.set_device(model_device)

    cfg_data["preproc"]["pipe"].remove("contours")

    if data_root is not None:
        cfg_data["dirs"].data_root = data_root.as_posix()

    if dump_path is None:
        dump_path = Path(cfg_data.dirs.data_root) / cfg_data.dirs.dump_folder

    cfg_data["parser"].dump_path = dump_path.as_posix()
    cfg_data["processor"]["dump"].data_root = cfg_data["dirs"].data_root
    cfg_data["processor"]["dump"].dump_path = dump_path.as_posix()

    if "SpeakerIDSetter" in cfg_data.get("singleton_handlers", {}):
        cfg_data["singleton_handlers"][
            "SpeakerIDSetter"
        ].resume_from_checkpoint = ckpt_path.as_posix()

    if "DatasetStatistics" in cfg_data.get("singleton_handlers", {}):
        cfg_data["singleton_handlers"][
            "DatasetStatistics"
        ].dump_path = dump_path.as_posix()

    if "StatisticsRange" in cfg_data.get("singleton_handlers", {}):
        cfg_data["singleton_handlers"]["StatisticsRange"].statistics_file = (
            dump_path / "ranges.json"
        ).as_posix()

    if "MeanBioEmbeddings" in cfg_data.get("singleton_handlers", {}):
        cfg_data["singleton_handlers"]["MeanBioEmbeddings"].mean_embeddings_file = (
            dump_path / "mean_bio_embeddings.json"
        ).as_posix()

    if textgrid_ext_old is not None:
        cfg_data["file_search"].ext = textgrid_ext_old

    if not dump_path.exists():
        print("Run dump")
        dump_path.mkdir(parents=True, exist_ok=True)
        config_path = dump_path / "tmp_cfg.yml"
        cfg_data.to_file(config_path)
        dump_worker(
            config_path,
            batch_size=batch_size,
            n_processes=n_processes,
            n_gpus=n_gpus,
            value_select=value_select,
        )

    if mapping_file and mapping_file.exists():
        print(f"Load labels mapping from {mapping_file.as_posix()}")
        mapping = np.load(mapping_file.as_posix())
    else:
        print("Run codebook clustering")
        codebook = model.encoder.vq_encoder.vq.codebook.weight.cpu().data.numpy()
        mapping = GaussianMixture(n_components=n_components).fit_predict(codebook)
        if mapping_file:
            np.save(mapping_file.as_posix(), mapping)

    print("Run dataset redefine")
    redefine_dataset(
        model=model,
        batch_processor=batch_processor,
        labels=mapping,
        config=cfg_data,
        value_select=value_select,
        batch_size=batch_size,
        n_processes=n_processes,
        textgrid_ext_new=textgrid_ext_new,
    )


if __name__ == "__main__":
    """
    Example: python -W ignore -m tts.acoustic_models.scripts.prepare_database
                --ckpt_path=/path/to/prosody/model
                --data_root=/path/to/data
                --dump_path=/path/to/dump
    """
    main(**parse_args().__dict__)
