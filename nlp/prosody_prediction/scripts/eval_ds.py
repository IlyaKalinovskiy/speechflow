from pathlib import Path

import torch

from tqdm import tqdm

from nlp import prosody_prediction
from nlp.prosody_prediction.callbacks import ProsodyCallback
from nlp.prosody_prediction.data_types import (
    ProsodyPredictionOutput,
    ProsodyPredictionTarget,
)
from speechflow.data_server.helpers import init_data_loader_from_config
from speechflow.training.saver import ExperimentSaver
from speechflow.training.utils.pad_utils import pad, pad_2d
from speechflow.utils.init import init_class_from_config


def eval(data_loader, model, callback, batch_processor, log_to_file=True):
    outputs_all = ProsodyPredictionOutput(
        binary=[],
        category=[],
    )
    targets_all = ProsodyPredictionTarget(
        binary=[],
        category=[],
    )
    texts = []

    with torch.no_grad():
        for batch_idx in tqdm(range(len(data_loader)), total=len(data_loader)):
            batch = data_loader.next_batch()
            inputs, targets, metadata = batch_processor(batch, batch_idx)
            outputs = model(inputs)
            if outputs.binary is not None:
                outputs_all.binary.extend(outputs.binary.to("cpu"))
                targets_all.binary.extend(targets.binary.to("cpu"))
            if outputs.category is not None:
                outputs_all.category.extend(outputs.category.to("cpu"))
                targets_all.category.extend(targets.category.to("cpu"))

            texts.append(callback._log_text(batch, outputs))

    if outputs_all.binary:
        outputs_all.binary, _ = pad_2d(outputs_all.binary, pad_id=0, height=2)
        targets_all.binary, _ = pad(targets_all.binary, pad_id=-100)
    if outputs_all.category:
        outputs_all.category, _ = pad_2d(
            outputs_all.category, pad_id=0, height=outputs_all.category[0].shape[1]
        )
        targets_all.category, _ = pad(targets_all.category, pad_id=-100)

    metrics, report = callback.compute_metrics(outputs_all, targets_all)

    if log_to_file:
        with open(
            "/src/experiments/wav_synt/report_ru.txt", "w", encoding="utf-8"
        ) as file:
            file.write("# REPORT\n\n")
            file.write("Binary\n\n")
            file.write(report["binary"])
            file.write("\n\nCategory\n\n")
            file.write(report["category"])
            file.write("\n\n## Metrics\n\n")
            file.write("\n\n".join([f"{m}={round(metrics[m], 3)}" for m in metrics]))
            file.write("\n\n## Texts\n\n")
            file.write("\n\n".join(texts))


if __name__ == "__main__":
    ckpt_path = "/src/experiments/10_Oct_2023_17_58_roberta_both/_checkpoints/epoch=14-step=7034.ckpt"
    device = torch.device("cpu")

    checkpoint = ExperimentSaver.load_checkpoint(Path(ckpt_path))
    cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)

    model_cls = getattr(prosody_prediction, cfg_model["net"]["type"])
    model = model_cls(checkpoint["params"])
    model.eval()

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    BatchProcessor = getattr(prosody_prediction, cfg_model["batch"]["type"])
    batch_processor = init_class_from_config(BatchProcessor, cfg_model["batch"])()
    batch_processor.device = device

    with init_data_loader_from_config(
        model_config_path="/src/nlp/prosody_prediction/configs/model.yml",
        data_config_path="/src/nlp/prosody_prediction/configs/data.yml",
        value_select=["en"],
        server_addr=None,
    ) as data_loaders:
        dl_train, dl_valid = data_loaders.values()
        callback = ProsodyCallback(
            data_loader=dl_valid,
            names=cfg_model["loss"]["names"],
            tokenizer=cfg_data["parser"]["tokenizer_name"],
            n_classes=cfg_model["net"]["params"]["n_classes"],
        )
        eval(dl_valid, model, callback, batch_processor)
