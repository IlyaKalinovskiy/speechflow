import typing as tp

from pathlib import Path

import torch

from multilingual_text_parser import Doc, TextParser

from nlp import pauses_prediction
from nlp.pauses_prediction.data_types import PausesPredictionOutput
from speechflow.data_pipeline.collate_functions.pauses_collate import (
    PausesPredictionCollateOutput,
)
from speechflow.data_pipeline.core import Batch, PipelineComponents
from speechflow.data_pipeline.datasample_processors.data_types import (
    PausesPredictionDataSample,
)
from speechflow.training.saver import ExperimentSaver
from speechflow.utils.dictutils import find_field
from speechflow.utils.init import init_class_from_config


class PausesPredictionInterface:
    def __init__(
        self,
        ckpt_path: tp.Union[str, Path],
        device: str = "cpu",
    ):
        checkpoint = ExperimentSaver.load_checkpoint(Path(ckpt_path))
        cfg_data, cfg_model = ExperimentSaver.load_configs_from_checkpoint(checkpoint)
        self.device = torch.device(device)

        model_cls = getattr(pauses_prediction, cfg_model["model"]["type"])
        self.model = model_cls(checkpoint["params"])
        self.model.eval()

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device)

        def _remove_item(_list: list, _name: str):
            if _name in _list:
                _list.remove(_name)

        _remove_item(cfg_data["preproc"]["pipe"], "add_pauses_from_timestamps")
        cfg_data["preproc"]["pipe"].insert(0, "add_pauses_from_text")
        cfg_data["preproc"]["add_pauses_from_text"] = {
            "level": "syntagmas",
            "num_symbols": 1,
        }

        self.pipeline = PipelineComponents(cfg_data, "valid")

        ignored_fields = {"word_timestamps", "phoneme_timestamps"}

        self.pipeline = self.pipeline.with_ignored_fields(
            ignored_data_fields=ignored_fields
        )

        BatchProcessor = getattr(pauses_prediction, cfg_model["batch"]["type"])
        self.batch_processor = init_class_from_config(
            BatchProcessor, cfg_model["batch"]
        )()
        self.batch_processor.device = self.device

        lang = find_field(cfg_data["preproc"], "lang")
        self.text_parser = TextParser(lang=lang)

    @staticmethod
    def _check_sils(
        batch: Batch,
        begin_pause: tp.Optional[float] = None,
        end_pause: tp.Optional[float] = None,
    ) -> bool:
        """
        :return: bool value indicating if there is at least one element in the batch
            with nontrivial sil token (i.e. pause duration that needs to be predicted).
            E.g: if there is no sil tokens inside a sentence and begin_pause and end_pause
            arguments are provided, there is no need to call pauses prediction model.
        """
        trivial_sil = False
        if begin_pause is not None and end_pause is not None:
            collated: PausesPredictionCollateOutput = batch.collated_samples  # type: ignore
            seq_lens = collated.transcription_lengths
            for i in range(len(batch)):
                if torch.sum(collated.sil_mask[i][1 : seq_lens[i] - 1]) == 0:  # type: ignore
                    trivial_sil = True
                    continue
                else:
                    trivial_sil = False
                    break
        return trivial_sil

    def _create_dummy_output(
        self, batch: Batch, begin_pause: float, end_pause: float
    ) -> PausesPredictionOutput:
        """For provided batch with only two sil tokens create `PausesPredictionOutput`
        with specified `begin_pause` and `end_pause` durations.

        Helper function that is used instead of pauses prediction model call.

        """
        assert batch.data_samples is not None
        default_durations = [
            [begin_pause] + [0.0] * (len(batch_sample.sil_mask) - 2) + [end_pause]
            for batch_sample in batch.data_samples
        ]

        collated: PausesPredictionCollateOutput = batch.collated_samples  # type: ignore
        default_durations = torch.Tensor(default_durations)

        return PausesPredictionOutput(
            sil_mask=collated.sil_mask,
            durations=default_durations.to(self.device),
        )

    @torch.inference_mode()
    def evaluate(self, batch: Batch) -> PausesPredictionOutput:
        inputs, _, _ = self.batch_processor(batch)
        outputs = self.model(inputs)
        return outputs

    def predict(
        self,
        text: tp.Union[Doc, str],
        speaker_id: int,
        begin_pause: tp.Optional[float] = None,  # type: ignore
        end_pause: tp.Optional[float] = None,  # type: ignore
    ) -> PausesPredictionOutput:
        """
        :param text:
        :param speaker_id:
        :param begin_pause: float
            pause duration in the beginning of the sentence.
        :param end_pause: float
            pause duration in the end of the sentence.
        """

        if isinstance(text, str):
            text = self.text_parser.process(Doc(text))

        samples = [
            PausesPredictionDataSample(sent=sent, speaker_id=speaker_id)
            for sent in text.sents
        ]

        batch = self.pipeline.datasample_to_batch(samples)
        collated: PausesPredictionCollateOutput = batch.collated_samples  # type: ignore

        # check if there are sil tokens inside sentence:
        trivial_sil = self._check_sils(batch, begin_pause, end_pause)
        if trivial_sil:
            outputs = self._create_dummy_output(batch, begin_pause, end_pause)  # type: ignore
        else:
            outputs = self.evaluate(batch)

        # assign begin and end pause duration if provided
        if not trivial_sil:
            if begin_pause is not None:
                outputs.durations[:, 0, :] = begin_pause
            if end_pause is not None:
                seq_lens = collated.transcription_lengths
                for i in range(len(batch)):
                    outputs.durations[i][seq_lens[i] - 1] = end_pause

        return outputs
