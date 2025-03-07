import typing as tp
import argparse

from pathlib import Path

import torch

from tqdm import tqdm

from speechflow.training.saver import ExperimentSaver
from speechflow.utils.fs import find_files


def prune(checkpoint: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    remove_keys = [
        "callbacks",
        "optimizer_states",
        "lr_schedulers",
        "scripts",
        "dataset",
    ]
    checkpoint = {k: v for k, v in checkpoint.items() if k not in remove_keys}

    remove_names = ["criterion", "discriminator_model"]
    state_dict = checkpoint["state_dict"]
    state_dict = {
        k: v for k, v in state_dict.items() if all(name not in k for name in remove_names)
    }
    checkpoint["state_dict"] = state_dict

    if "lang_id_map" in checkpoint:
        print("langs:", list(checkpoint["lang_id_map"].keys()))

    if "speaker_id_map" in checkpoint:
        print("speakers:", list(checkpoint["speaker_id_map"].keys()))

    return checkpoint


def prune_checkpoint(
    ckpt_dir: Path,
    state_dict_only: bool = False,
    overwrite: bool = False,
):
    if ckpt_dir.is_file():
        pathes = [ckpt_dir]
    else:
        pathes = find_files(ckpt_dir.as_posix(), extensions=(".ckpt",))

    for path in tqdm(pathes):
        ckpt_path = Path(path)
        print(ckpt_path.as_posix())

        if overwrite:
            out_ckpt_path = ckpt_path
        else:
            out_ckpt_path = ckpt_path.with_suffix(".pt")

        checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)
        checkpoint = prune(checkpoint)

        if state_dict_only:
            checkpoint = {"state_dict": checkpoint["state_dict"]}

        print("checkpoint keys:", list(checkpoint.keys()))

        torch.save(checkpoint, out_ckpt_path)
        print(out_ckpt_path.as_posix())


if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-d", "--ckpt_dir", help="checkpoints directory", type=Path, required=True
    )
    arguments_parser.add_argument(
        "-st",
        "--state_dict_only",
        help="remove all without state_dict",
        type=bool,
        default=False,
    )
    arguments_parser.add_argument(
        "--overwrite", help="overwrite checkpoints", type=bool, default=False
    )
    args = arguments_parser.parse_args()

    prune_checkpoint(args.ckpt_dir, args.state_dict_only, args.overwrite)
