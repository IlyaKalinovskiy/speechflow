import sys
import pickle
import typing as tp
import logging

from pathlib import Path

import pytorch_lightning as pl

from pytorch_lightning import seed_everything

# required for multi-gpu training
THIS_PATH = Path(__file__).absolute()
ROOT = THIS_PATH.parents[3]
sys.path.append(ROOT.as_posix())

from speechflow.data_pipeline.datasample_processors.tts_text_processors import (
    TTSTextProcessor,
)
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.data_server.loader import DataLoader
from speechflow.io import Config, tp_PATH, tp_PATH_LIST
from speechflow.logging import trace
from speechflow.logging.server import LoggingServer
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.config_prepare import model_config_prepare, train_arguments
from speechflow.utils.init import init_class_from_config
from speechflow.utils.profiler import Profiler
from tts import vocoders
from tts.vocoders import vocos

LOGGER = logging.getLogger("root")


def update_vocos_model_config(cfg: Config, dl: DataLoader):
    speaker_id_handler = dl.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        cfg["model"]["feature_extractor"][
            "init_args"
        ].n_langs = speaker_id_handler.n_langs
        cfg["model"]["feature_extractor"][
            "init_args"
        ].n_speakers = speaker_id_handler.n_speakers
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    return cfg, lang_id_map, speaker_id_map


def train(cfg_model: Config, data_loaders: tp.Dict[str, DataLoader]) -> str:
    experiment_path = Path(cfg_model["experiment_path"])

    seed_everything(cfg_model.get("seed"))

    dl_train, dl_valid = data_loaders.values()

    batch_processor_cls = getattr(vocoders, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(batch_processor_cls, cfg_model["batch"])()

    speaker_id_handler = dl_train.client.find_info("SpeakerIDSetter")
    if speaker_id_handler is not None:
        lang_id_map = speaker_id_handler.lang2id
        speaker_id_map = speaker_id_handler.speaker2id
    else:
        lang_id_map = None
        speaker_id_map = None

    saver = ExperimentSaver(
        expr_path=experiment_path,
        additional_files={
            "data.yml": dl_train.client.info["data_config_raw"],
            "model.yml": cfg_model.raw_file,
        },
    )
    saver.to_save.update(
        {
            "lang_id_map": lang_id_map,
            "speaker_id_map": speaker_id_map,
        }
    )

    if 1:
        saver.to_save["info"] = dl_train.client.info
    else:
        info_file_name = f"{experiment_path.name}_info.pkl"
        info_file_path = experiment_path / info_file_name
        info_file_path.write_bytes(pickle.dumps(dl_train.client.info))
        saver.to_save["files"]["info_file_name"] = info_file_name

    expr_name = cfg_model["experiment"].class_name
    if "Vocos" in expr_name:
        pl_engine_cls = getattr(vocos, expr_name)

        feat_cfg = cfg_model["model"].feature_extractor

        if feat_cfg.class_name == "TTSFeatures":
            lang = dl_train.client.find_info("lang", "RU")
            text_proc = TTSTextProcessor(lang=lang)
            feat_cfg.init_args["tts_cfg"].alphabet_size = text_proc.alphabet_size
            feat_cfg.init_args[
                "tts_cfg"
            ].n_symbols_per_token = text_proc.num_symbols_per_phoneme_token
            feat_cfg.init_args["tts_cfg"].n_langs = speaker_id_handler.n_langs
            feat_cfg.init_args["tts_cfg"].n_speakers = speaker_id_handler.n_speakers
        else:
            feat_cfg.init_args.n_langs = speaker_id_handler.n_langs
            feat_cfg.init_args.n_speakers = speaker_id_handler.n_speakers

        feat_cls = getattr(vocos, feat_cfg.class_name)
        feat = init_class_from_config(feat_cls, feat_cfg.init_args)()

        backbone_cfg = cfg_model["model"].backbone
        backbone_cls = getattr(vocos, backbone_cfg.class_name)
        backbone = init_class_from_config(backbone_cls, backbone_cfg.init_args)()

        head_cfg = cfg_model["model"].head
        head_cls = getattr(vocos, head_cfg.class_name)
        head = init_class_from_config(head_cls, head_cfg.init_args)()

        pl_engine = init_class_from_config(
            pl_engine_cls, cfg_model["experiment"].init_args
        )(feat, backbone, head, batch_processor, saver)
    else:
        raise NotImplementedError

    callbacks = [
        saver.get_checkpoint_callback(cfg=cfg_model["checkpoint"]),
    ]

    if cfg_model.get("callbacks"):
        for callback_name, callback_cfg in cfg_model["callbacks"].items():
            if hasattr(vocoders.callbacks, callback_name):
                cls = getattr(vocoders.callbacks, callback_name)
            else:
                cls = getattr(pl.callbacks, callback_name)
            callback = init_class_from_config(cls, callback_cfg)()
            callbacks.append(callback)

    ckpt_path = cfg_model["trainer"].pop("resume_from_checkpoint", None)

    trainer = init_class_from_config(pl.Trainer, cfg_model["trainer"])(
        default_root_dir=experiment_path, callbacks=callbacks
    )

    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(pl_engine, dl_train, dl_valid, ckpt_path=ckpt_path)

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
    cfg_model = model_config_prepare(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        value_select=value_select,
        resume_from=resume_from,
        expr_suffix=expr_suffix,
    )

    with LoggingServer.ctx(cfg_model["experiment_path"]):
        with init_data_loader_from_config(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            value_select=value_select,
            server_addr=data_server_address,
        ) as data_loaders:
            try:
                return train(cfg_model=cfg_model, data_loaders=data_loaders)
            except Exception as e:
                LOGGER.error(trace("main", e))
                raise e


if __name__ == "__main__":
    """
    example:
        python -W ignore train.py -c=configs/vocos/mel_vocoder.yml -cd=configs/vocos/mel_data_24khz.yml
        python -W ignore train.py -c=configs/vocos/mel_vocoder.yml -cd=configs/vocos/mel_data_24khz.yml -vs debug

        # When training on multiple GPUs you need to set the flag NCCL_P2P_DISABLE:

        NCCL_P2P_DISABLE=1 python -W ignore train.py

    """
    args = train_arguments().parse_args()
    print(main(**args.__dict__))
