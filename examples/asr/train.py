import typing as tp

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio
import pytorch_lightning as pl

from pytorch_lightning import Callback
from torch import nn
from torch.nn import functional as F

from speechflow.data_pipeline.collate_functions.utils import collate_sequence
from speechflow.data_pipeline.core import (
    BaseBatchProcessor,
    BaseCollate,
    BaseCollateOutput,
    BaseDSParser,
    BaseDSProcessor,
    Batch,
    DataPipeline,
    DataSample,
    PipeRegistry,
    TrainData,
    tp_DATA,
)
from speechflow.data_pipeline.samplers import RandomSampler
from speechflow.data_server.helpers import LoaderParams, init_data_loader
from speechflow.data_server.loader import DataLoader
from speechflow.io import (
    AudioChunk,
    Config,
    check_path,
    construct_file_list,
    json_load_from_file,
    split_file_list,
)
from speechflow.logging import set_verbose_logging
from speechflow.logging.server import LoggingServer
from speechflow.training import (
    BaseCriterion,
    BaseTorchModel,
    BaseTorchModelParams,
    ExperimentSaver,
    LightningEngine,
    Optimizer,
)
from speechflow.utils.init import init_class_from_config, lazy_initialization
from speechflow.utils.profiler import Profiler
from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence


#
# --------------  DATA DESCRIPTION --------------
#


@dataclass
class MorseDataSample(DataSample):
    audio_chunk: AudioChunk = None
    waveform: tp_DATA = None
    mel: tp_DATA = None
    transcription: tp_DATA = None
    tokens: tp_DATA = None


@dataclass
class MorseCollateOutput(MorseDataSample, BaseCollateOutput):
    waveform_lengths: tp_DATA = None
    mel_lengths: tp_DATA = None
    token_lengths: tp_DATA = None


@dataclass
class MorseTarget(TrainData):
    transcription: tp_DATA = None
    tokens: tp_DATA = None
    token_lengths: tp_DATA = None


@dataclass
class MorseForwardInput(TrainData):
    waveform: tp_DATA = None
    waveform_lengths: tp_DATA = None
    mel: tp_DATA = None
    mel_lengths: tp_DATA = None


@dataclass
class MorseForwardOutput(TrainData):
    logits: tp_DATA = None
    log_probs: tp_DATA = None
    output_lengths: tp_DATA = None


#
# -------------- DATA PROCESSORS --------------
#


class DSParser(BaseDSParser):
    def reader(self, file_path: Path, label=None) -> tp.List[tp.Dict[str, tp.Any]]:
        audio_chunk = AudioChunk(file_path.with_suffix(".opus"))
        transcription = file_path.with_suffix(".txt").read_text(encoding="utf-8")
        metadata = {
            "file_path": file_path,
            "audio_chunk": audio_chunk,
            "transcription": transcription.strip(),
        }
        return [metadata]

    def converter(self, metadata: tp.Dict[str, tp.Any]) -> tp.List[MorseDataSample]:
        ds = MorseDataSample(
            file_path=metadata.get("file_path"),
            audio_chunk=metadata.get("audio_chunk"),
            transcription=metadata.get("transcription"),
        )
        return [ds]


class SignalProcessor(BaseDSProcessor):
    def __init__(self, sample_rate: int = 8000):
        super().__init__()
        self._sample_rate = sample_rate

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"audio_chunk", "waveform"})
    def process(self, ds: MorseDataSample) -> MorseDataSample:
        ds.audio_chunk.load(sr=self._sample_rate)
        ds.waveform = ds.audio_chunk.waveform  # type: ignore
        return ds


class MelProcessor(BaseDSProcessor):
    def __init__(
        self,
        sample_rate: int = 8000,
        n_fft: int = 512,
        hop_len: int = 128,
        win_len: int = 512,
        n_mels: int = 80,
        **kwargs,
    ):
        super().__init__()

        self._mel_cfg = self.get_config_from_locals(
            ignore=["win_len", "hop_len", "n_mels"]
        )
        self._mel_cfg["window_size"] = win_len / sample_rate
        self._mel_cfg["window_stride"] = hop_len / sample_rate
        self._mel_cfg["features"] = n_mels

        self._mel_module = None

    def init(self):
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

        super().init()
        self._mel_module = AudioToMelSpectrogramPreprocessor(**self._mel_cfg)
        self._mel_module.eval()

    @PipeRegistry.registry(inputs={"audio_chunk"}, outputs={"mel"})
    @lazy_initialization
    def process(self, ds: MorseDataSample) -> MorseDataSample:
        waveform = ds.audio_chunk.waveform
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform_length = torch.LongTensor([waveform.shape[-1]])
        mel, _ = self._mel_module.get_features(
            input_signal=waveform, length=waveform_length
        )
        ds.mel = mel.squeeze(0).T
        return ds.to_numpy()


class MorseTokenizer(BaseDSProcessor):
    @check_path(assert_file_exists=True)
    def __init__(
        self,
        alphabet_path: str | Path,
        blank_symbol: str = "_",
        use_morse_alphabet: bool = False,
        use_morse_correction: bool = False,
    ):
        super().__init__()
        self.blank_symbol = blank_symbol

        self.alphabet = {self.blank_symbol: None}
        self.alphabet.update(json_load_from_file(alphabet_path))

        self.symbol_to_id = {s: i for i, s in enumerate(self.alphabet)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.alphabet)}

        self.morse_alphabet = [blank_symbol, "|", "-", ".", " "]
        self.morse_symbol_to_id = {s: i for i, s in enumerate(self.morse_alphabet)}
        self.morse_id_to_symbol = {i: s for i, s in enumerate(self.morse_alphabet)}

        self.use_morse_alphabet = use_morse_alphabet
        self.use_morse_correction = use_morse_correction
        self.logging_params(self.get_config_from_locals())

    @property
    def labels(self) -> tp.List[str]:
        if not self.use_morse_alphabet:
            return list(self.id_to_symbol.values())
        else:
            return list(self.morse_id_to_symbol.values())

    @property
    def blank_index(self) -> int:
        if not self.use_morse_alphabet:
            return self.symbol_to_id[self.blank_symbol]
        else:
            return self.morse_symbol_to_id[self.blank_symbol]

    @PipeRegistry.registry(inputs={"transcription"}, outputs={"tokens"})
    def process(self, ds: MorseDataSample) -> MorseDataSample:
        if not self.use_morse_alphabet:
            ds.tokens = np.asarray(
                [self.symbol_to_id[t] for t in ds.transcription], dtype=np.int64
            )
        else:
            morse_transcription = []
            for s in ds.transcription:
                for m in self.alphabet[s] if self.alphabet[s] is not None else s:
                    morse_transcription.append(m)
                morse_transcription.append("|")
            morse_transcription.pop()

            ds.tokens = np.asarray(
                [self.morse_symbol_to_id[t] for t in morse_transcription], dtype=np.int64
            )
        return ds

    def morse_to_char(self, morse_word: str) -> str:
        if morse_word == "":
            return ""
        elif morse_word == " ":
            return " "
        else:
            try:
                return {v: k for k, v in self.alphabet.items()}[morse_word]
            except KeyError:
                if self.use_morse_correction:
                    for k, v in self.alphabet.items():
                        if v and torchaudio.functional.edit_distance(v, morse_word) == 1:
                            return k

                return ""


class MorseCollate(BaseCollate):
    def collate(self, batch: tp.List[MorseDataSample]) -> MorseCollateOutput:  # type: ignore
        collated = super().collate(batch)
        collated = MorseCollateOutput(**collated.to_dict())  # type: ignore

        collated.waveform, collated.waveform_lengths = collate_sequence(
            batch, "waveform", pad_values=0
        )
        # collated.mel, collated.mel_lengths = collate_sequence(batch, "mel", pad_values=0)
        collated.tokens, collated.token_lengths = collate_sequence(
            batch, "tokens", pad_values=0
        )

        collated.transcription = [ds.transcription for ds in batch]  # type: ignore
        return collated


#
# -------------- MODEL --------------
#


class MorseModelParams(BaseTorchModelParams):
    output_dim: int = 256
    with_rnn: bool = True


class MorseModel(BaseTorchModel):
    params: MorseModelParams

    def __init__(
        self,
        cfg: tp.Union[Config, MorseModelParams],
        strict_init: bool = True,
    ):
        super().__init__(MorseModelParams.create(cfg, strict_init))

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
        self.model = bundle.get_model()

        inner_dim = self.model.model.aux.in_features
        self.proj = nn.Linear(inner_dim, self.params.output_dim)

        if self.params.with_rnn:
            self.rnn = nn.GRU(
                inner_dim,
                inner_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = None

    def forward(self, inputs: MorseForwardInput) -> MorseForwardOutput:
        x_lens = (inputs.waveform_lengths * inputs.waveform.shape[1]).long()
        x, _ = self.model.model.feature_extractor(inputs.waveform, x_lens)

        x = self.model.model.encoder(x)

        if self.params.with_rnn:
            x_lens = (inputs.waveform_lengths * x.shape[1]).long()
            y = run_rnn_on_padded_sequence(self.rnn, x, x_lens)
        else:
            y = x

        z = self.proj(y)

        return MorseForwardOutput(
            logits=z,
            log_probs=F.log_softmax(z, dim=-1),
            output_lengths=inputs.waveform_lengths,
        )


class BatchProcessor(BaseBatchProcessor):
    def __call__(
        self, batch: Batch, batch_idx: int = 0, global_step: int = 0
    ) -> (MorseForwardInput, MorseTarget, tp.List[MorseDataSample]):
        _collated: MorseCollateOutput = batch.collated_samples  # type: ignore

        _input: MorseForwardInput = init_class_from_config(
            MorseForwardInput, _collated.to_dict(), check_keys=False
        )()
        _target: MorseTarget = init_class_from_config(
            MorseTarget, _collated.to_dict(), check_keys=False
        )()

        return _input.to(self.device), _target.to(self.device), batch.data_samples


#
# -------------- CRITERION AND CALLBACKS --------------
#


class Criterion(BaseCriterion):
    def __init__(self, blank_index: int = 0):
        super().__init__()
        self.blank_index = blank_index

    def forward(
        self,
        output: MorseForwardOutput,
        target: MorseTarget,
        batch_idx: int = 0,
        global_step: int = 0,
    ) -> tp.Dict[str, torch.Tensor]:
        input_lens = (output.output_lengths * output.log_probs.shape[1]).ceil().long()
        target_lens = (target.token_lengths * target.tokens.shape[1]).ceil().long()
        ctc_loss = torch.nn.functional.ctc_loss(
            output.log_probs.transpose(0, 1),
            target.tokens.int(),
            input_lens,
            target_lens,
            self.blank_index,
            zero_infinity=True,
            reduction="mean",
        )
        return {"ctc_loss": ctc_loss}


class ValidationMetrics(Callback):
    def __init__(self, tokenizer: MorseTokenizer):
        self.tokenizer = tokenizer
        self.decoder = GreedyCTCDecoder(tokenizer.labels, tokenizer.blank_index)
        self.metrics = None

    def on_validation_start(self, pl_module: pl.LightningModule, *args):
        self.metrics = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        _, outputs, targets, _ = pl_module.test_step(batch, batch_idx)

        for i in range(outputs.logits.shape[0]):
            actual_transcript = targets.transcription[i]
            greedy_result = self.decoder(outputs.logits[i])

            if self.tokenizer.use_morse_alphabet:
                chars = []
                for word in greedy_result.split("|"):
                    chars.append(self.tokenizer.morse_to_char(word))

                predict_transcript = "".join(chars)
            else:
                predict_transcript = greedy_result

            greedy_cer = torchaudio.functional.edit_distance(
                actual_transcript, predict_transcript
            )
            self.metrics["CER"].append(greedy_cer)

            if batch_idx == 0 and pl_module.logger is not None:
                pl_module.logger.experiment.add_text(
                    f"transcription_{i} <actual>|<predict>",
                    f"<{actual_transcript}>|<{predict_transcript}>",
                    global_step=pl_module.global_step,
                )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
    ):
        pl_module.log("CER", np.asarray(self.metrics["CER"]).mean(), prog_bar=True)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank_index: int = 0):
        super().__init__()
        self.labels = labels
        self.blank_index = blank_index

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_index]
        joined = "".join([self.labels[i] for i in indices])
        return joined.strip()


#
# -------------- TRAIN LOOP AND EVALUATION --------------
#


def create_data_pipline(subsets, alphabet_path):
    set_verbose_logging()

    parser = DSParser()
    preprocessing = [
        SignalProcessor().process,
        # MelProcessor().process,
        MorseTokenizer(alphabet_path=alphabet_path).process,
    ]
    collate = MorseCollate(relative_lengths=True)
    sampler = RandomSampler()

    return DataPipeline.init_from_components(
        subsets, parser, preprocessing, collate, sampler, alphabet_path=alphabet_path
    )


def train(
    experiment_path: str | Path,
    loaders: tp.Dict[str, DataLoader],
    accelerator: str = "cpu",
    max_epochs: int = 20,
):
    # set seed
    pl.seed_everything(1234)

    dl_train, dl_valid = loaders.values()

    # initialize batch processor
    batch_processor = BatchProcessor()

    # create dnn model
    tokenizer = MorseTokenizer(dl_train.client.find_info("alphabet_path"))

    model_params = MorseModelParams(output_dim=len(tokenizer.labels))
    model = MorseModel(model_params)

    # create criterion
    criterion = Criterion(blank_index=tokenizer.blank_index)

    # create optimizer
    optimizer = Optimizer(
        model,
        method={"type": "AdamW", "weight_decay": 1.0e-6},
        lr_scheduler={"type": "WarmupInvRsqrtLR", "lr_max": 0.001},
    )

    # create experiment saver
    saver = ExperimentSaver(
        expr_path=experiment_path,
    )

    # create engine
    pl_engine: LightningEngine = LightningEngine(
        model=model,
        criterion=criterion,
        batch_processor=batch_processor,
        optimizer=optimizer,
        saver=saver,
    )

    # create trainer callbacks
    checkpoint_callback_cfg = Config(
        {
            "monitor": "ctc_loss/valid",
            "filename": "morse_{epoch}_{step}_{CER:.4f}",
            "mode": "min",
            "save_top_k": 1,
            "every_n_epochs": 1,
        }
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        saver.get_checkpoint_callback(cfg=checkpoint_callback_cfg),
        ValidationMetrics(tokenizer),
    ]

    # create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        default_root_dir=experiment_path,
        callbacks=callbacks,
    )

    # lets try to train
    with Profiler("training", format=Profiler.Format.h):
        trainer.fit(pl_engine, dl_train, dl_valid)

    print(f"Model training completed!\nExperiment path: {experiment_path}")


@check_path(assert_file_exists=True)
def test(expr_path: str | Path, file_path: str | Path, alphabet_path: str | Path):
    ckpt_path = ExperimentSaver.get_last_checkpoint(expr_path)
    if ckpt_path is None:
        raise FileNotFoundError("checkpoint not found")

    checkpoint = ExperimentSaver.load_checkpoint(ckpt_path)

    model = MorseModel(checkpoint["params"])
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])

    tokenizer = MorseTokenizer(alphabet_path)
    greedy_decoder = GreedyCTCDecoder(tokenizer.labels, tokenizer.blank_index)

    metadata = _data_pipeline["valid"].dataset_parser.reader(file_path)
    batch = _data_pipeline["valid"].metadata_to_batch(metadata)

    with torch.inference_mode():
        model_input, _, _ = BatchProcessor()(batch)
        predict: MorseForwardOutput = model(model_input)

    greedy_result = greedy_decoder(predict.logits[0])

    if tokenizer.use_morse_alphabet:
        chars = []
        for word in greedy_result.split("|"):
            chars.append(tokenizer.morse_to_char(word))

        predict_transcript = "".join(chars)
    else:
        predict_transcript = greedy_result

    print("target: ", metadata[0]["transcription"])
    print("predict:", predict_transcript)


if __name__ == "__main__":
    # Perform training data generation use https://github.com/1-800-BAD-CODE/MorseCodeToolkit
    # Alphabet file format (alphabet.json):
    #
    #   {
    #       " ": null
    #       "А": ".-",
    #       "Б": "-...",
    #       ...
    #   }
    #

    _dataset_path = r"cw_dataset_internship_rus_20k"
    _alphabet_path = r"alphabet.json"
    _expr_path = "_logs/test_morse_code_recognition"

    _data_pipeline = create_data_pipline(
        subsets=["train", "valid"], alphabet_path=_alphabet_path
    )

    _flist = construct_file_list(_dataset_path, ext=".txt", with_subfolders=True)
    _flist_train, _flist_valid = split_file_list(_flist, ratio=0.8)

    if 1:  # TRAINING
        with LoggingServer.ctx(_expr_path):
            with init_data_loader(
                loader_params=LoaderParams(batch_size=16, non_stop=True),
                data_pipeline=_data_pipeline,
                flist_by_subsets={"train": _flist_train, "valid": _flist_valid},
                n_processes=8,
            ) as _loaders:
                train(_expr_path, _loaders)
    else:  # EVALUATION
        test(_expr_path, _flist[-1], _alphabet_path)
