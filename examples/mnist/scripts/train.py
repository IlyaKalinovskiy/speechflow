import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from examples import mnist
from examples.mnist.callbacks import AccuracyCallback
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training import ExperimentSaver, LightningEngine, Optimizer
from speechflow.training.utils.config_prepare import model_config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler

LOGGER = logging.getLogger("root")


def train(model_cfg: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(model_cfg["experiment_path"])

    # set seed
    seed_everything(model_cfg.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    # initialize batch processor
    batch_processor_cls = getattr(mnist, model_cfg["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, model_cfg["batch"])()

    # create dnn model
    model_cls = getattr(mnist, model_cfg["net"]["type"])
    model = init_class_from_config(model_cls, model_cfg["net"]["params"])()

    # create criterion
    criterion_cls = getattr(mnist, model_cfg["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, model_cfg["loss"])()

    # create optimizer
    optimizer = init_class_from_config(Optimizer, model_cfg["optimizer"])(model=model)

    # create experiment saver
    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": model_cfg.raw_file,
        },
    )

    # create engine
    net_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    # create trainer callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        AccuracyCallback(dl_valid),
        saver.get_checkpoint_callback(cfg=model_cfg["checkpoint"]),
    ]

    ckpt_path = model_cfg["trainer"].pop("resume_from_checkpoint", None)

    # create trainer
    trainer = init_class_from_config(pl.Trainer, model_cfg["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

    # lets try to train
    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(net_engine, dl_train, dl_valid, ckpt_path=ckpt_path)

    LOGGER.info("Model training completed!")
    return experiment_path.as_posix()


def main(
    model_config_path: tp_PATH,
    data_config_path: tp.Optional[tp.Union[tp_PATH, tp_PATH_LIST]] = None,
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
                return train(model_cfg=model_cfg, data_loaders=data_loaders)
            except Exception as e:
                LOGGER.error(trace("main", e))
                raise e


if __name__ == "__main__":
    """
    example:
        train.py -c=../configs/lenet.yml -cd=../configs/data.yml
        train.py -c=../configs/resnet.yml -cd=../configs/data.yml

    """
    args = train_arguments().parse_args()

    if not Path("data").exists():
        from examples.mnist.scripts.data_prepare import prepare_data

        prepare_data()

    print(main(**args.__dict__))
