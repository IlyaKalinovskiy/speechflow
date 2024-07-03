import typing as tp

from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.data_pipeline.collate_functions.spectrogram_collate import (
    SpectrogramCollate,
    SpectrogramCollateOutput,
)
from speechflow.data_pipeline.core.datasample import ToNumpy, ToTensor
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodySSMLDataSample,
    TTSDataSample,
)
from speechflow.training.utils.pad_utils import pad, pad_2d, sequence_collate

__all__ = [
    "TTSCollate",
    "TTSCollateOutput",
    "TTSCollateWithPrompt",
    "TTSCollateOutputWithPrompt",
    "TTSCollateWithSSML",
    "TTSCollateOutputWithSSML",
    "LinguisticFeatures",
]


@dataclass
class LinguisticFeatures(ToTensor, ToNumpy):
    pos_tags: torch.LongTensor = None
    punctuation: torch.LongTensor = None
    token_ends: torch.LongTensor = None
    syntagma_ends: torch.LongTensor = None
    syntax: torch.LongTensor = None
    syntax_importance: torch.FloatTensor = None
    emphasis: torch.LongTensor = None
    intonation: torch.LongTensor = None
    breath_mask: torch.FloatTensor = None
    prosody: torch.LongTensor = None
    sil_mask: torch.BoolTensor = None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def num_integer_features() -> int:
        return str(LinguisticFeatures.__annotations__).count("torch.LongTensor")

    @staticmethod
    def num_float_features() -> int:
        return str(LinguisticFeatures.__annotations__).count("torch.FloatTensor")

    @staticmethod
    def collate(batch: tp.List[TTSDataSample]) -> "LinguisticFeatures":
        pad_symb_id = batch[0].pad_symb_id
        batch_feat = {}
        if batch[0].ling_feat is not None:
            for seq_name in batch[0].ling_feat.keys():
                sequence = [sample.ling_feat.get(seq_name) for sample in batch]
                sequence, input_lens = pad(sequence, pad_symb_id)

                if seq_name == "sil_mask":
                    sequence = sequence.bool()

                batch_feat[seq_name] = sequence

        return LinguisticFeatures(**batch_feat)


@dataclass
class TTSCollateOutput(SpectrogramCollateOutput):
    transcription: Tensor = None
    transcription_by_frames: Tensor = None
    transcription_lengths: Tensor = None
    transcription_by_frames_lengths: Tensor = None
    ling_feat: LinguisticFeatures = None  # type: ignore
    lm_feat: Tensor = None
    durations: Tensor = None
    invert_durations: Tensor = None
    aggregated: tp.Dict[str, Tensor] = None  # type: ignore
    synt_lengths: Tensor = None
    num_words: Tensor = None
    word_lengths: Tensor = None
    num_tokens: Tensor = None
    token_lengths: Tensor = None
    concatenate: Tensor = None

    def __post_init__(self):
        super().__post_init__()

        if self.aggregated is None:
            self.aggregated = {}


@dataclass
class TTSCollateOutputWithPrompt(TTSCollateOutput):
    prompt: tp.Optional[TTSCollateOutput] = None


@dataclass
class TTSCollateOutputWithSSML(TTSCollateOutput):
    pitch_modifier: Tensor = None
    volume_modifier: Tensor = None
    temp_modifier: Tensor = None


class TTSCollate(SpectrogramCollate):
    def __call__(self, batch: tp.List[TTSDataSample]) -> TTSCollateOutput:  # type: ignore
        spec_collated = super().__call__(batch)  # type: ignore
        collated = TTSCollateOutput(**spec_collated.to_dict())  # type: ignore
        multiple = self.multiple.get("spec")

        pad_symb_id = batch[0].pad_symb_id
        sil_symb_id = batch[0].sil_symb_id

        transcription, input_lens = sequence_collate(batch, "transcription", pad_symb_id)
        transcription_by_frames, frames_input_lens = sequence_collate(
            batch, "transcription_by_frames", sil_symb_id
        )
        lm_feat, _ = sequence_collate(batch, "lm_feat")
        durations, _ = sequence_collate(batch, "durations")
        invert_durations, _ = sequence_collate(
            batch, "invert_durations", multiple=multiple
        )
        words_lens, num_words = sequence_collate(batch, "word_lengths", pad_symb_id)
        syntagmas_lens, _ = sequence_collate(batch, "synt_lengths", pad_symb_id)

        if batch[0].aggregated is not None:
            aggregated = {}
            for name in batch[0].aggregated.keys():
                data = [sample.aggregated[name] for sample in batch]
                pad_id = batch[0].get_param_val("mel_min_val") if "mel" in name else 0.0
                aggregated[name], _ = (
                    pad(data)
                    if data[0].ndim == 1
                    else pad_2d(data, pad_id, data[0].shape[1])
                )
        else:
            aggregated = {}

        if batch[0].concatenate is not None:
            concatenate = [sample.concatenate for sample in batch]
            concatenate, _ = pad_2d(concatenate, n_channel=concatenate[0].shape[1])
        else:
            concatenate = None

        collated.transcription = transcription
        collated.transcription_lengths = input_lens
        collated.transcription_by_frames = transcription_by_frames
        collated.transcription_by_frames_lengths = frames_input_lens
        collated.ling_feat = LinguisticFeatures.collate(batch)
        collated.lm_feat = lm_feat
        collated.durations = durations
        collated.invert_durations = invert_durations
        collated.aggregated = aggregated
        collated.concatenate = concatenate
        collated.num_words = num_words
        collated.word_lengths = words_lens
        collated.num_tokens = input_lens
        collated.token_lengths = input_lens
        collated.synt_lengths = syntagmas_lens
        return collated


class TTSCollateWithPrompt(TTSCollate):
    def __call__(self, batch: tp.List[TTSDataSample]) -> TTSCollateOutputWithPrompt:  # type: ignore
        idx_neighbor = [x.additional_fields["neighbor_idx"] for x in batch]
        idx_neighbor = torch.Tensor(idx_neighbor).long()
        idx_right: tp.Iterable = torch.where(idx_neighbor == torch.roll(idx_neighbor, 1))[0]  # type: ignore
        idx_left: tp.Iterable = idx_right - 1  # type: ignore

        batch_prompt = [batch[n] for n in idx_left]
        batch_target = [batch[n] for n in idx_right]
        batch_tts_collated_prompt = super().__call__(batch_prompt)
        batch_tts_collated_target = super().__call__(batch_target)

        collated: TTSCollateOutputWithPrompt = TTSCollateOutputWithPrompt(
            **vars(batch_tts_collated_target), prompt=batch_tts_collated_prompt
        )

        return collated


class TTSCollateWithSSML(TTSCollate):
    def __call__(self, batch: tp.List[ProsodySSMLDataSample]) -> TTSCollateOutputWithSSML:  # type: ignore
        tts_collated = super().__call__(batch)  # type: ignore
        collated = TTSCollateOutputWithSSML(**tts_collated.to_dict())  # type: ignore

        temp_modifier, _ = sequence_collate(batch, "temp_modifier", 1.0)
        volume_modifier, _ = sequence_collate(batch, "volume_modifier", 1.0)
        pitch_modifier, _ = sequence_collate(batch, "pitch_modifier", 1.0)

        collated.temp_modifier = temp_modifier
        collated.volume_modifier = volume_modifier
        collated.pitch_modifier = pitch_modifier
        return collated


if __name__ == "__main__":

    print("num_integer_ling_features:", LinguisticFeatures.num_integer_features())
    print("num_float_ling_features:", LinguisticFeatures.num_float_features())
