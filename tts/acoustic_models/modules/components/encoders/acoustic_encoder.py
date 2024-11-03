import torch

from torch import nn
from torch.nn import functional as F

from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.layers import Conv
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput
from tts.acoustic_models.modules.params import EncoderParams

__all__ = [
    "AcousticEncoder",
    "AcousticEncoderParams",
    "AcousticEncoderWithClassificationAdaptor",
    "AcousticEncoderWithClassificationAdaptorParams",
]


class AcousticEncoderParams(EncoderParams):
    upsample: bool = False

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class AcousticEncoder(Component):
    params: AcousticEncoderParams

    def __init__(self, params: AcousticEncoderParams, input_dim: int):
        super().__init__(params, input_dim)
        inner_dim = params.encoder_inner_dim

        self.prenet = PreNet(input_dim, inner_dim, inner_dim)

        self.convs = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(inner_dim),
            nn.ConvTranspose1d(inner_dim, inner_dim, 4, 2, 1)
            if params.upsample
            else nn.Identity(),
            nn.Conv1d(inner_dim, inner_dim, 5, 1, 2),
            nn.ReLU(),
            nn.InstanceNorm1d(inner_dim),
        )

        if params.use_projection and inner_dim != params.encoder_output_dim:
            self.proj = Regression(
                inner_dim,
                params.encoder_output_dim,
                p_dropout=params.projection_p_dropout,
                activation_fn=params.projection_activation_fn,
            )
        else:
            self.proj = nn.Identity()

    @property
    def output_dim(self):
        if self.params.use_projection:
            return self.params.encoder_output_dim
        else:
            return self.params.encoder_inner_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        x = self.prenet(x)
        z = self.convs(x.transpose(1, -1)).transpose(1, -1)
        y = self.proj(z)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(z)


class PreNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AcousticEncoderWithClassificationAdaptorParams(AcousticEncoderParams):
    n_convolutions: int = 3
    kernel_size: int = 5


class AcousticEncoderWithClassificationAdaptor(AcousticEncoder):
    params: AcousticEncoderWithClassificationAdaptorParams

    def __init__(
        self, params: AcousticEncoderWithClassificationAdaptorParams, input_dim: int
    ):
        super().__init__(params, input_dim)

        convolutions = []
        for _ in range(params.n_convolutions):
            conv_layer = nn.Sequential(
                Conv(
                    self.output_dim,
                    self.output_dim,
                    kernel_size=params.kernel_size,
                    stride=1,
                    padding=int((params.kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(self.output_dim),
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.components_output_dim["adaptor_context"] = lambda: self.output_dim

    def forward_step(self, x: ComponentInput) -> EncoderOutput:
        result: EncoderOutput = super().forward_step(x)

        content = result.content
        adaptor_context = result.additional_content.setdefault(
            f"adaptor_context_{self.id}", []
        )

        if self.training:
            ctx = content.transpose(2, 1)
            for conv in self.convolutions:
                ctx = F.relu(conv(ctx))

            adaptor_context.append(ctx.transpose(2, 1))

        return result
