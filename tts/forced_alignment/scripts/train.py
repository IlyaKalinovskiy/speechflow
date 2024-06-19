import sys
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())

from speechflow.data_pipeline.datasample_processors import TextProcessor
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training import ExperimentSaver, LightningEngine, Optimizer
from speechflow.training.utils.config_prepare import model_config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import forced_alignment
from tts.forced_alignment.callbacks import AligningVisualisationCallback

LOGGER = logging.getLogger("root")


def train(model_cfg: Config, data_loaders: tp.Dict[str, DataLoader]):
    experiment_path = Path(model_cfg["experiment_path"])

    pl.seed_everything(model_cfg.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(forced_alignment, model_cfg["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, model_cfg["batch"])()

    lang = dl_train.client.find_info("lang")
    text_proc = TextProcessor(lang=lang)
    model_cfg["net"]["params"].n_symbols = text_proc.alphabet_size
    model_cfg["net"][
        "params"
    ].n_symbols_per_token = text_proc.num_symbols_per_phoneme_token

    hop_len = dl_train.client.find_info("hop_len")
    sr = dl_train.client.find_info("sample_rate")
    model_cfg["net"]["params"].frames_per_sec = sr / hop_len

    speaker_id_handler = dl_train.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        model_cfg["net"]["params"].n_langs = speaker_id_handler.n_langs
        model_cfg["net"]["params"].n_speakers = speaker_id_handler.n_speakers
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    model_cls = getattr(forced_alignment, model_cfg["net"]["type"])
    model = init_class_from_config(model_cls, model_cfg["net"]["params"])()

    if "init_from" in model_cfg["net"]:
        ckpt_path = Path(model_cfg["net"]["init_from"].get("ckpt_path", ""))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} is not found")

        LOGGER.warning(f"Loading {ckpt_path.as_posix()}")
        ckpt = ExperimentSaver.load_checkpoint(ckpt_path)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

        if model_cfg["net"]["params"].n_langs < len(ckpt["lang_id_map"]):
            model_cfg["net"]["params"].n_langs = len(ckpt["lang_id_map"])
        if model_cfg["net"]["params"].n_speakers < len(ckpt["speaker_id_map"]):
            model_cfg["net"]["params"].n_speakers = len(ckpt["speaker_id_map"])

        try:
            model = init_class_from_config(model_cls, model_cfg["net"]["params"])()
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            LOGGER.error(trace("train", e))
            remove_modules = model_cfg["net"]["init_from"].get(
                "remove_modules",
                [
                    "embedding",
                    "encoder",
                    "phoneme_proj",
                    "ling_feat_proj",
                    "lang_emb",
                    "cond_proj",
                ],
            )
            if model_cfg["net"]["params"].n_langs != len(ckpt["lang_id_map"]):
                remove_modules.append("lang_emb")

            state_dict = {
                k: v
                for k, v in state_dict.items()
                if all(m not in k for m in remove_modules)
            }
            model.load_state_dict(state_dict, strict=False)

    criterion_cls = getattr(forced_alignment, model_cfg["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, model_cfg["loss"])()

    optimizer = init_class_from_config(Optimizer, model_cfg["optimizer"])(model=model)

    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": model_cfg.raw_file,
        },
    )
    saver.to_save.update(
        {
            "lang_id_map": lang_id_map,
            "speaker_id_map": speaker_id_map,
            "alphabet": text_proc.alphabet,
        }
    )
    saver.to_save.update({"dataset": dl_train.client.info["dataset"]})

    net_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        AligningVisualisationCallback(),
        saver.get_checkpoint_callback(cfg=model_cfg["checkpoint"]),
    ]

    if "early_stopping" in model_cfg:
        early_stop_callback = init_class_from_config(
            pl.callbacks.EarlyStopping, model_cfg["early_stopping"]
        )()
        callbacks.append(early_stop_callback)

    ckpt_path = model_cfg["trainer"].pop("resume_from_checkpoint", None)

    trainer = init_class_from_config(pl.Trainer, model_cfg["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(net_engine, dl_train, dl_valid, ckpt_path=ckpt_path)

    LOGGER.info("Model training completed!")
    return experiment_path.as_posix()


def main(
    model_config_path: Path,
    data_config_path: tp.Optional[Path] = None,
    resume_from: tp.Optional[Path] = None,
    data_server_address: tp.Optional[str] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    expr_suffix: tp.Optional[str] = None,
):
    model_cfg = model_config_prepare(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        resume_from=resume_from,
        value_select=value_select,
        expr_suffix=expr_suffix,
    )

    with LoggingServer.ctx(model_cfg["experiment_path"]):
        with init_data_loader_from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            value_select=value_select,
            server_addr=data_server_address,
        ) as data_loaders:
            try:
                return train(model_cfg=model_cfg, data_loaders=data_loaders)
            except Exception as e:
                LOGGER.error(trace("main", e))
                raise e


if __name__ == "__main__":
    """
    example:
        train.py -c=../configs/2stage/model_stage1.yml -cd=../configs/2stage/data_stage1.yml
        train.py -c=../configs/2stage/model_stage2.yml -cd=../configs/2stage/data_stage2.yml

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
