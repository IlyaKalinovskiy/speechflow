import typing as tp

from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.data_pipeline.core.base_collate_fn import BaseCollate, BaseCollateOutput
from speechflow.data_pipeline.datasample_processors.data_types import ImageDataSample

__all__ = ["ImageCollate", "ImageCollateOutput"]


@dataclass
class ImageCollateOutput(BaseCollateOutput):
    images: tp.Optional[Tensor] = None
    labels: tp.Optional[Tensor] = None


class ImageCollate(BaseCollate):
    def __call__(self, batch: tp.List[ImageDataSample]) -> ImageCollateOutput:  # type: ignore
        collated = super().__call__(batch)  # type: ignore
        collated = ImageCollateOutput(**collated.to_dict())  # type: ignore

        images = torch.cat([ds.image for ds in batch])
        labels = torch.cat([torch.LongTensor([ds.label]) for ds in batch])

        collated.images = images
        collated.labels = labels
        return collated
