import typing as tp

import torch

from torch import nn
from torch.nn.utils import weight_norm

from speechflow.training.base_model import BaseTorchModelParams
from tts.vocoders.vocos.modules.backbones.base import Backbone
from tts.vocoders.vocos.modules.backbones.blocks import ResBlock1

__all__ = ["VocosResNetBackbone", "VocosResNetBackboneParams"]


class VocosResNetBackboneParams(BaseTorchModelParams):
    input_dim: int
    inner_dim: int
    num_layers: int
    layer_scale_init_value: tp.Optional[float] = None


class VocosResNetBackbone(Backbone):
    params: VocosResNetBackboneParams

    """Vocos backbones module built with ResBlocks.

    Args:
        input_dim (int): Number of input features channels.
        inner_dim (int): Hidden dimension of the model.
        num_layers (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.

    """

    def __init__(self, params: VocosResNetBackboneParams):
        super().__init__(params)

        self.embed = weight_norm(
            nn.Conv1d(params.input_dim, params.inner_dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = (
            params.layer_scale_init_value or 1 / params.num_layers / 3
        )
        self.resnet = nn.Sequential(
            *[
                ResBlock1(
                    dim=params.inner_dim, layer_scale_init_value=layer_scale_init_value
                )
                for _ in range(params.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
