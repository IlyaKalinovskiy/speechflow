import sys
import pickle
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())

from speechflow.data_pipeline.datasample_processors import TextProcessor
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training.lightning_trainer import LightningEngine
from speechflow.training.optimizer import Optimizer
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import model_config_prepare, train_arguments
from speechflow.training.utils.finetuning import prepare_finetuning, prepare_warmstart
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import acoustic_models

LOGGER = logging.getLogger("root")


def update_model_config(cfg: Config, dl: DataLoader):
    lang = dl.client.find_info("lang", "RU")
    text_proc = TextProcessor(lang=lang)
    cfg["net"]["params"].n_symbols = text_proc.alphabet_size
    cfg["net"]["params"].n_symbols_per_token = text_proc.num_symbols_per_phoneme_token

    ds_stat = dl.client.find_info("DatasetStatistics")
    if ds_stat:
        sr = dl.client.find_info("sample_rate")
        hop_len = dl.client.find_info("hop_len")
        cfg["net"]["params"].setdefault(
            "max_input_length", int(ds_stat.max_transcription_length * 1.1)
        )
        cfg["net"]["params"].setdefault(
            "max_output_length", int(ds_stat.max_wave_duration * sr / hop_len * 1.1)
        )

    speaker_id_handler = dl.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        cfg["net"]["params"].n_langs = speaker_id_handler.n_langs
        cfg["net"]["params"].n_speakers = speaker_id_handler.n_speakers
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    return cfg, lang_id_map, speaker_id_map, text_proc.alphabet


def train(model_cfg: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(model_cfg["experiment_path"])

    seed_everything(model_cfg.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(acoustic_models, model_cfg["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, model_cfg["batch"])()

    model_cfg, lang_id_map, speaker_id_map, alphabet = update_model_config(
        model_cfg, dl_train
    )

    model_cls = getattr(acoustic_models, model_cfg["net"]["type"])

    if hasattr(model_cls, "update_and_validate_model_params"):
        model_cfg = model_cls.update_and_validate_model_params(
            model_cfg,
            dl_train.client.info["data_config"],
        )

    if model_cfg.get("finetuning") is not None:
        model = prepare_finetuning(
            model_cls, model_cfg["finetuning"], model_cfg["net"]["params"]
        )
    else:
        model = init_class_from_config(model_cls, model_cfg["net"]["params"])()

    if model_cfg.get("warmstart") is not None:
        model = prepare_warmstart(model, model_cfg["warmstart"])

    criterion_cls = getattr(acoustic_models, model_cfg["loss"]["type"])
    criterion = init_class_from_config(criterion_cls, model_cfg["loss"])()

    optimizer = init_class_from_config(Optimizer, model_cfg["optimizer"])(
        model=model, criterion=criterion
    )

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
            "alphabet": alphabet,
        }
    )

    ckpt_path = experiment_path / f"{experiment_path.name}_info.pkl"
    ckpt_path.write_bytes(pickle.dumps(dl_train.client.info))

    net_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    callbacks = [
        saver.get_checkpoint_callback(cfg=model_cfg["checkpoint"]),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if model_cfg.get("callbacks"):
        for callback_name, callback_cfg in model_cfg["callbacks"].items():
            cls = getattr(acoustic_models.callbacks, callback_name)
            callback = init_class_from_config(cls, callback_cfg)()
            callbacks.append(callback)

    ckpt_path = model_cfg["trainer"].pop("resume_from_checkpoint", None)

    trainer = init_class_from_config(pl.Trainer, model_cfg["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

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
        python -W ignore train.py -c=../configs/tts/tts_forward.yml -cd=../configs/tts/tts_data_24khz.yml
        python -W ignore train.py -c=../configs/tts/tts_forward.yml -cd=../configs/tts/tts_data_24khz.yml -vs debug

        # When training on multiple GPUs you need to set the flag NCCL_P2P_DISABLE:

        NCCL_P2P_DISABLE=1 python -W ignore train.py

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
