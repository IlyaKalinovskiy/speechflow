from dataclasses import dataclass

from torch import Tensor

from speechflow.data_pipeline.core import TrainData

__all__ = [
    "MNISTTarget",
    "MNISTForwardInput",
    "MNISTForwardOutput",
]


@dataclass
class MNISTTarget(TrainData):
    labels: Tensor = None


@dataclass
class MNISTForwardInput(TrainData):
    images: Tensor = None


@dataclass
class MNISTForwardOutput(TrainData):
    logits: Tensor = None
