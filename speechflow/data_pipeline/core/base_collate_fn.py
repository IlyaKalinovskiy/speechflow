import typing as tp
import logging
import numbers

from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.data_pipeline.core.datasample import (
    DataSample,
    Detachable,
    MovableToDevice,
    Pinnable,
    ToDict,
    ToTensor,
)
from speechflow.training.utils.pad_utils import pad, pad_2d

__all__ = ["BaseCollate", "BaseCollateOutput", "NoDataSamplesError"]

LOGGER = logging.getLogger("root")


class NoDataSamplesError(Exception):
    pass


@dataclass
class BaseCollateOutput(ToDict, ToTensor, MovableToDevice, Pinnable, Detachable):
    label: Tensor = None
    extra_info: tp.Dict[str, tp.Any] = None
    additional_fields: tp.Dict[str, tp.Any] = None

    def __post_init__(self):
        if self.extra_info is None:
            self.extra_info = {}
        if self.additional_fields is None:
            self.additional_fields = {}


class BaseCollate:
    def __init__(
        self,
        pad_values: tp.Optional[tp.Dict[str, float]] = None,
        multiple: tp.Optional[tp.Dict[str, float]] = None,
        additional_fields: tp.Optional[tp.List[str]] = None,
    ):
        self.pad_values = pad_values if pad_values else {}
        self.multiple = multiple if multiple else {}
        self.additional_fields = additional_fields if additional_fields else []

    def __call__(self, batch: tp.List[DataSample]) -> BaseCollateOutput:
        if len(batch) == 0:
            raise NoDataSamplesError("No DataSamples in batch")

        multiple = self.multiple.get("spec")

        for ds in batch:
            ds.deserialize(full=True).to_tensor()

        collated = BaseCollateOutput()

        if batch[0].label is not None:
            labels = [sample.label for sample in batch]
            if isinstance(labels[0], numbers.Number):
                labels = torch.Tensor(labels)
        else:
            labels = None

        collated.extra_info = {}
        for field in self.additional_fields:
            collated.extra_info.update({field: [getattr(b, field) for b in batch]})

        additional_fields = {}
        if batch[0].additional_fields:
            for key in batch[0].additional_fields.keys():
                values = [sample.additional_fields[key] for sample in batch]

                if "mel" in key:
                    pad_id = batch[0].get_param_val("mel_min_val")
                else:
                    pad_id = 0.0

                if values[0].ndim == 1:
                    if "dura" in key:
                        values, _ = pad(values, pad_id)
                    else:
                        values, _ = pad(values, pad_id, multiple=multiple)
                else:
                    values, _ = pad_2d(
                        values,
                        pad_id,
                        n_channel=values[0].shape[1],
                        multiple=multiple,
                    )

                additional_fields[key] = values

        collated.label = labels
        collated.additional_fields = additional_fields
        return collated
