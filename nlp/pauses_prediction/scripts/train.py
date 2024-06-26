import sys
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())

from nlp import pauses_prediction
from speechflow.data_pipeline.datasample_processors import TextProcessor
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training.lightning_trainer import LightningEngine
from speechflow.training.optimizer import Optimizer
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import model_config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler

LOGGER = logging.getLogger("root")


def train(cfg: dict, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(cfg["experiment_path"])

    seed_everything(cfg.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(pauses_prediction, cfg["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg["batch"])()

    lang = dl_train.client.find_info("lang")
    cfg["model"]["params"]["n_symbols"] = TextProcessor(lang=lang).alphabet_size

    speaker_id_handler = dl_train.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        cfg["model"]["params"]["n_speakers"] = speaker_id_handler.n_speakers
        speaker_map = speaker_id_handler.speaker2id
    else:
        speaker_map = None

    model_cls = getattr(pauses_prediction, cfg["model"]["type"])
    model = init_class_from_config(model_cls, cfg["model"]["params"])()

    criterion_cls = getattr(pauses_prediction, cfg["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, cfg["loss"])()

    optimizer = init_class_from_config(Optimizer, cfg["optimizer"])(model=model)

    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": cfg["model_config_raw"],
        },
    )
    saver.to_save.update({"speaker_id_map": speaker_map})
    saver.to_save.update({"dataset": dl_train.client.info["dataset"]})

    pl_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    callbacks = [
        saver.get_checkpoint_callback(cfg=cfg["checkpoint"]),
    ]

    ckpt_path = cfg["trainer"].pop("resume_from_checkpoint", None)

    trainer = init_class_from_config(pl.Trainer, cfg["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(pl_engine, dl_train, dl_valid, ckpt_path=ckpt_path)

    LOGGER.info("Model training completed!")
    return experiment_path.as_posix()


def main(
    model_config_path: Path,
    data_config_path: tp.Optional[tp.Union[Path, tp.List[Path]]] = None,
    value_select: tp.Optional[tp.List[str]] = None,
    resume_from: tp.Optional[Path] = None,
    data_server_address: tp.Optional[str] = None,
    expr_suffix: tp.Optional[str] = None,
) -> str:
    model_cfg = model_config_prepare(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        value_select=value_select,
        resume_from=resume_from,
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
                return train(cfg=model_cfg, data_loaders=data_loaders)
            except Exception as e:
                LOGGER.error(trace("main", e))
                raise e


if __name__ == "__main__":
    """
    example:
        train.py -c=../configs/wavernn/model.yml -cd=../configs/wavernn/data.yml

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
