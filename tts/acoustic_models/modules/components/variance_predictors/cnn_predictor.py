import typing as tp

import torch

from pydantic import Field
from torch import nn

from speechflow.utils.tensor_utils import apply_mask, get_mask_from_lengths
from tts.acoustic_models.modules.common.layers import Conv, LearnableSwish
from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = ["CNNPredictor", "CNNPredictorParams"]


class CNNPredictorParams(VariancePredictorParams):
    kernel_sizes: tp.Tuple[int, ...] = (3, 7, 13, 3)
    dropout: float = 0.1
    as_encoder: bool = False
    var_params: tp.Dict[str, tp.Any] = Field(default_factory=lambda: {})


class CNNPredictor(Component):
    params: CNNPredictorParams

    def __init__(
        self, params: CNNPredictorParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)

        first_convs_kernel_sizes = params.kernel_sizes[:-1]
        second_convs_kernel_sizes = params.kernel_sizes[-1]

        self.first_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        input_dim,
                        params.vp_inner_dim,
                        kernel_size=k,
                        padding=(k - 1) // 2,
                        w_init_gain=None,
                        swap_channel_dim=True,
                    ),
                    LearnableSwish(),
                    nn.LayerNorm(params.vp_inner_dim),
                    nn.Dropout(params.dropout),
                )
                for k in first_convs_kernel_sizes
            ]
        )

        self.second_conv = nn.Sequential(
            Conv(
                params.vp_inner_dim * len(first_convs_kernel_sizes),
                params.vp_inner_dim,
                kernel_size=second_convs_kernel_sizes,
                padding=(second_convs_kernel_sizes - 1) // 2,
                w_init_gain=None,
                swap_channel_dim=True,
            ),
            LearnableSwish(),
            nn.LayerNorm(params.vp_inner_dim),
            nn.Dropout(params.dropout),
        )

        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(params.vp_inner_dim, params.vp_inner_dim),
            nn.ReLU(),
            nn.Linear(params.vp_inner_dim, params.vp_output_dim),
        )

    @property
    def output_dim(self):
        return (
            self.params.vp_output_dim
            if self.params.as_encoder
            else self.params.vp_inner_dim
        )

    def encode(self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        after_first_conv = []
        for conv_layer in self.first_convs:
            after_first_conv.append(conv_layer(x))

        concatenated = torch.cat(after_first_conv, dim=2)
        after_second_conv = self.second_conv(concatenated)

        for conv_1 in after_first_conv:
            after_second_conv += conv_1

        output = after_second_conv
        return apply_mask(output, get_mask_from_lengths(x_lengths))

    def decode(self, encoder_output, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs):
        output = self.predictor(encoder_output)
        output = output.squeeze(-1)
        return apply_mask(output, get_mask_from_lengths(x_lengths))

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        precompute_name = f"imputer_{kwargs.get('name')}_precompute"
        if precompute_name in kwargs["model_inputs"].additional_inputs:
            return kwargs["model_inputs"].additional_inputs[precompute_name], {}, {}

        encoder_output = self.encode(x, x_lengths, model_inputs, **kwargs)

        if not self.params.as_encoder:
            out = self.decode(encoder_output, x_lengths, model_inputs, **kwargs)
            return out, {}, {}
        else:
            return encoder_output, {}, {}
