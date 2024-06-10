import typing as tp

from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.core.base_collate_fn import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import (
    ProsodyPredictionDataSample,
)
from speechflow.training.utils.pad_utils import pad

__all__ = ["ProsodyPredictionCollate", "ProsodyPredictionCollateOutput"]


@dataclass
class ProsodyPredictionCollateOutput(BaseCollateOutput):
    attention_mask: tp.Optional[Tensor] = None
    input_ids: tp.Optional[Tensor] = None
    binary: tp.Optional[Tensor] = None
    category: tp.Optional[Tensor] = None


class ProsodyPredictionCollate(BaseCollate):
    def __call__(  # type: ignore
        self, batch: tp.List[ProsodyPredictionDataSample]
    ) -> ProsodyPredictionCollateOutput:
        collated = super().__call__(batch)  # type: ignore
        collated = ProsodyPredictionCollateOutput(**collated.to_dict())  # type: ignore

        pad_symb_id = batch[0].pad_id

        binary = []
        category = []
        attention_mask = []
        input_ids = []

        for sample in batch:
            binary.append(sample.binary)
            category.append(sample.category)
            attention_mask.append(sample.attention_mask)
            input_ids.append(sample.input_ids)

        if batch[0].binary is not None:
            binary, _ = pad(binary, pad_val=-100)
        else:
            binary = None
        if batch[0].category is not None:
            category, _ = pad(category, pad_val=-100)
        else:
            category = None

        input_ids, _ = pad(input_ids, pad_val=pad_symb_id)
        attention_mask, _ = pad(attention_mask, pad_val=0)

        collated.attention_mask = attention_mask
        collated.input_ids = input_ids
        collated.binary = binary
        collated.category = category
        return collated
