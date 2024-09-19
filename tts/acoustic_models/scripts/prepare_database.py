import typing as tp
import logging
import argparse
import warnings

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from speechflow.data_server.helpers import dataset_iterator
from speechflow.io import AudioSeg, Config
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.gpu_info import get_freer_gpu
from speechflow.utils.init import init_class_from_config
from tts import acoustic_models

LOGGER = logging.getLogger("root")

warnings.filterwarnings("ignore", category=UserWarning, module="russian_g2p")


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-ckpt", "--model_checkpoint", help="model checkpoint", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-d", "--data_root", help="data root", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-dump",
        "--dump_folder",
        help="path to data dump",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-ranges",
        "--ranges_file_path",
        help="path to ranges.json",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-embeds",
        "--mean_embeddings_file_path",
        help="path to mean_bio_embeddings.json",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-cd",
        "--config_data_path",
        help="path to yaml configs for data",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-vs", "--value_select", help="select specific values", type=str, nargs="+"
    )
    arguments_parser.add_argument(
        "-eps", "--eps", help="eps for DBSCAN", type=float, default=3
    )
    arguments_parser.add_argument(
        "-n", "--n_components", help="number of prosodic classes", type=int, default=10
    )
    arguments_parser.add_argument(
        "-d", "--device", help="device", type=str, default="cpu"
    )
    arguments_parser.add_argument(
        "-nproc",
        "--n_processes",
        help="number of workers for data processing",
        type=int,
        default=1,
    )
    arguments_parser.add_argument(
        "-m", "--mapping_file", help="file to store mapping", type=Path, default=None
    )
    arguments_parser.add_argument(
        "-ft",
        "--filename_text",
        help="file to store dataset for text prediction",
        type=Path,
        default=None,
    )
    arguments_parser.add_argument(
        "-cf", "--codebook_file", help="file to store codebook", type=Path, default=None
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


def clustering(embeddings: np.array, eps: int = 6, n_components: int = 10):
    """
    Function for clustering of embeddings from the codebook
    1. Outliers are obtained by DBSCAN
    2. Outliers are clustered by GaussianMixture clustering
    3. Indices from the codebook are mapped with the corresponding classes
    """

    classes = DBSCAN(eps=eps, min_samples=2).fit_predict(embeddings)
    outliers = embeddings[np.where(classes == -1)]
    if outliers.shape[0] == 0:
        print("No outliers found. Reduce eps")

    classes_gmm = GaussianMixture(n_components=n_components).fit_predict(outliers)
    out = np.where(classes == -1)[0]

    mapping = []
    for x in range(embeddings.shape[0]):
        if x in out:
            mapping.append(classes_gmm[np.where(out == x)[0][0]])
        else:
            mapping.append(-1)
    mapping = np.array(mapping)

    return mapping


def redefine_segs(
    model,
    batch_processor,
    labels: np.array,
    filename_text: tp.Optional[Path] = None,
    config_data_path: tp.Optional[Path] = None,
    data_config: tp.Optional[Config] = None,
    vs: tp.Optional[tp.List[str]] = None,
    device: str = "cpu",
    n_processes: int = 1,
    textgrid_ext_new: str = ".TextGrid_prosody",
):
    """Function that adds additional layer with prosody to segs."""

    if filename_text is not None:
        df = {
            "token": [],
            "index": [],
            "set": [],
            "pos": [],
            "syntax": [],
            "emphasis": [],
            "speaker": [],
            "lang": [],
            "punct": [],
        }
    iterator = dataset_iterator(
        config_data_path=config_data_path,
        config_data=data_config,
        value_select=vs,
        device=device,
        n_processes=n_processes,
    )

    number_of_classes = defaultdict(int)
    total_number_of_words = 0

    for batch in tqdm(iterator, total=len(iterator)):
        if batch:
            for ds in batch.data_samples:
                if not ds:
                    continue
                tg_path = ds.file_path
                if tg_path.with_suffix(textgrid_ext_new).exists():
                    continue

                model_inputs, _, _ = batch_processor(batch)

                with torch.no_grad():
                    out = model(model_inputs)

                indices_ = out.additional_content["encoding_indices_encoder_0"]

                sega = AudioSeg.load(tg_path)
                if sega is None:
                    continue

                tokens = [token.text for token in ds.sent.tokens if token.pos != "PUNCT"]
                indices_ = [
                    labels[int(i.item())]
                    for i, t in zip(indices_, tokens)
                    if t not in ["<BOS>", "<EOS>", "<SIL>"]
                ]

                j = 0
                for i, token in enumerate(sega.sent.tokens):
                    if token.pos != "PUNCT":
                        ind = str(indices_[j])
                        sega.sent.tokens[i].prosody = ind
                        j += 1
                        punct = "no"
                        total_number_of_words += 1
                        number_of_classes[ind] += 1
                    else:
                        sega.sent.tokens[i].prosody = "-1"
                        punct = token.text

                    if filename_text is not None:
                        df["set"].append(str(tg_path))
                        df["token"].append(token.text)
                        df["index"].append(sega.sent.tokens[i].prosody)
                        df["pos"].append(token.pos)
                        df["syntax"].append(token.rel)
                        df["emphasis"].append(token.emphasis)
                        df["speaker"].append(sega.meta["speaker_name"])
                        df["lang"].append(sega.meta["lang"])
                        if i != 0:
                            df["punct"].append(punct)

                if filename_text is not None:
                    df["punct"].append("no")

                sega.meta["source_sega"] = tg_path.name
                sega.save(tg_path.with_suffix(textgrid_ext_new))

    print("Percent of each class and its absolute number:")
    print(
        "\n".join(
            [
                f"{cl}: {round(percentage/total_number_of_words, 3)*100}% ({percentage})"
                for cl, percentage in sorted(number_of_classes.items())
            ]
        )
    )

    if filename_text is not None:
        df = pd.DataFrame(df)
        df.to_csv(filename_text.as_posix())


def main(
    model_checkpoint: Path,
    data_root: Path,
    dump_folder: Path = None,
    ranges_file_path: Path = None,
    mean_embeddings_file_path: Path = None,
    config_data_path: Path = None,
    filename_text: Path = None,
    value_select: tp.Optional[tp.List[str]] = None,
    device: str = "cpu",
    n_processes: int = 1,
    eps: int = 6,
    n_components: int = 10,
    mapping_file: tp.Optional[Path] = None,
    codebook_file: tp.Optional[Path] = None,
    textgrid_ext_old: tp.Optional[str] = None,
    textgrid_ext_new: str = ".TextGrid_prosody",
):
    checkpoint = ExperimentSaver.load_checkpoint(model_checkpoint)
    data_config, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

    if device == "cuda":
        model_device = torch.device(f"cuda:{get_freer_gpu()}")
    else:
        model_device = torch.device("cpu")

    model_cls = getattr(acoustic_models, cfg_model["model"]["type"])
    model = model_cls(checkpoint["params"])
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(model_device)

    batch_processor_cls = getattr(acoustic_models, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()
    batch_processor.device = model_device

    data_config["dirs"]["data_root"] = data_root
    data_config["singleton_handlers"]["SpeakerIDSetter"][
        "resume_from_checkpoint"
    ] = model_checkpoint.as_posix()
    data_config["singleton_handlers"]["SpeakerIDSetter"][
        "remove_unknown_speakers"
    ] = False
    data_config["preproc"].pop("contours")
    data_config["preproc"]["pipe"].remove("contours")

    if dump_folder is not None and (
        dump_folder.exists() or (data_config["dirs"]["data_root"] / dump_folder).exists()
    ):
        data_config["dirs"]["dump_folder"] = dump_folder.as_posix()
        data_config["parser"]["dump"] = dump_folder.as_posix()
        data_config["processor"]["dump"]["folder_path"] = dump_folder.as_posix()
        data_config["singleton_handlers"]["DatasetStatistics"][
            "dump"
        ] = dump_folder.as_posix()
        data_config["singleton_handlers"]["StatisticsRange"]["statistics_file"] = (
            dump_folder / "ranges.json"
        ).as_posix()
        data_config["singleton_handlers"]["SpeakerIDSetter"]["mean_embeddings_file"] = (
            dump_folder / "mean_bio_embeddings.json"
        ).as_posix()
    elif ranges_file_path is not None and ranges_file_path.exists():
        data_config["parser"]["dump"] = None
        data_config["processor"]["dump"] = None
        data_config["singleton_handlers"]["DatasetStatistics"]["dump"] = None
        data_config["singleton_handlers"]["StatisticsRange"][
            "statistics_file"
        ] = ranges_file_path.as_posix()
        if mean_embeddings_file_path is not None and mean_embeddings_file_path.exists():
            data_config["singleton_handlers"]["SpeakerIDSetter"][
                "mean_embeddings_file"
            ] = mean_embeddings_file_path.as_posix()
    else:
        raise ValueError(f"Dump folder {dump_folder} does not exist!")

    if textgrid_ext_old is not None:
        data_config["file_search"]["ext"] = textgrid_ext_old

    if mapping_file and mapping_file.exists():
        print(f"Load labels mapping from {mapping_file.as_posix()}")
        mapping = np.load(mapping_file.as_posix())
    else:
        print("Run codebook clustering")
        codebook = model.encoder.encoder.vq._embedding.weight
        embeddings = codebook.cpu().detach()
        if codebook_file:
            torch.save(embeddings, codebook_file)

        mapping = clustering(embeddings=embeddings, eps=eps, n_components=n_components)
        if mapping_file:
            np.save(mapping_file.as_posix(), mapping)

    redefine_segs(
        model=model,
        batch_processor=batch_processor,
        config_data_path=config_data_path,
        data_config=data_config,
        labels=mapping,
        vs=value_select,
        device=device,
        filename_text=filename_text,
        n_processes=n_processes,
        textgrid_ext_new=textgrid_ext_new,
    )


if __name__ == "__main__":
    """
    Example: python -W ignore -m tts.acoustic_models.scripts.prepare_database
                --model_checkpoint=/path/to/prosody/model
                --data_path=/path/to/data
                --dump_folder=/path/to/dump
    """
    main(**parse_args().__dict__)
