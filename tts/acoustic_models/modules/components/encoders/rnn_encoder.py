import typing as tp

import torch

from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common import AdaLayerNorm
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["RNNEncoder", "RNNEncoderParams"]


class RNNEncoderParams(CNNEncoderParams):
    # condition
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Literal["cat", "adanorm"] = "cat"

    # rnn
    rnn_bidirectional: bool = True
    rnn_p_dropout: float = 0.1

    # projection
    use_projection: bool = True
    projection_p_dropout: float = 0.1
    projection_activation_fn: str = "Identity"


class RNNEncoder(CNNEncoder):
    params: RNNEncoderParams

    def __init__(self, params: RNNEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim

        if not params.condition:
            self.rnn = nn.LSTM(
                in_dim,
                params.encoder_inner_dim // (params.rnn_bidirectional + 1),
                num_layers=params.encoder_num_layers,
                bidirectional=params.rnn_bidirectional,
                dropout=params.rnn_p_dropout,
                batch_first=True,
            )
        elif params.condition_type == "cat":
            self.rnns = nn.ModuleList()
            for i in range(params.encoder_num_layers):
                if i == 0:
                    in_dim = in_dim + params.condition_dim
                else:
                    in_dim = params.encoder_inner_dim + params.condition_dim

                self.rnns.append(
                    nn.LSTM(
                        in_dim,
                        params.encoder_inner_dim // (params.rnn_bidirectional + 1),
                        num_layers=1,
                        bidirectional=params.rnn_bidirectional,
                        dropout=params.rnn_p_dropout,
                        batch_first=True,
                    )
                )
        elif params.condition_type == "adanorm":
            in_dim += params.condition_dim

            self.rnns = nn.ModuleList()
            for i in range(params.encoder_num_layers):
                if i == 0:
                    self.rnns.append(
                        nn.LSTM(
                            in_dim,
                            params.encoder_inner_dim // (params.rnn_bidirectional + 1),
                            num_layers=1,
                            bidirectional=params.rnn_bidirectional,
                            dropout=params.rnn_p_dropout,
                            batch_first=True,
                        )
                    )
                else:
                    self.rnns.append(
                        nn.LSTM(
                            params.encoder_inner_dim + params.condition_dim,
                            params.encoder_inner_dim // (params.rnn_bidirectional + 1),
                            num_layers=1,
                            bidirectional=params.rnn_bidirectional,
                            dropout=params.rnn_p_dropout,
                            batch_first=True,
                        )
                    )

                if i + 1 != params.encoder_num_layers:
                    self.rnns.append(
                        AdaLayerNorm(params.encoder_inner_dim, params.condition_dim)
                    )

        if (
            params.use_projection
            and params.encoder_inner_dim != params.encoder_output_dim
        ):
            self.proj = Regression(
                params.encoder_inner_dim,
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
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        if not self.params.condition:
            x = run_rnn_on_padded_sequence(self.rnn, x, x_lens)
        else:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)

            s = cond.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)

            for block in self.rnns:
                if isinstance(block, AdaLayerNorm):
                    x = block(x, cond)
                    continue

                x = torch.cat([x, s], dim=2)
                x = run_rnn_on_padded_sequence(block, x, x_lens)

        y = self.proj(x)

        return EncoderOutput.copy_from(inputs).set_content(y).set_hidden_state(x)
