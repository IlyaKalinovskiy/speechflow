import torch

from torch import nn

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.backbones.base import Backbone

__all__ = ["DummyBackbone", "DummyBackboneParams"]


class DummyBackboneParams(BaseTorchModelParams):
    input_dim: int
    inner_dim: int


class DummyBackbone(Backbone):
    params: DummyBackboneParams

    def __init__(self, params: DummyBackboneParams):
        super().__init__(params)
        if params.input_dim != params.inner_dim:
            self.proj = nn.Linear(params.input_dim, params.inner_dim)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.proj is not None:
            y = self.proj(x)
        else:
            y = x

        return y.transpose(1, -1)
