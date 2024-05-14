import typing as tp

import torch

from torch import nn

from speechflow.training.utils.tensor_utils import apply_mask, run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.blocks import Regression
from tts.acoustic_models.modules.common.layer_norm import AdaLayerNorm
from tts.acoustic_models.modules.components.encoders.cnn_encoder import (
    CNNEncoder,
    CNNEncoderParams,
)
from tts.acoustic_models.modules.data_types import ComponentInput, EncoderOutput

__all__ = ["RNNEncoder", "RNNEncoderParams"]


class RNNEncoderParams(CNNEncoderParams):
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0
    condition_type: tp.Literal["cat", "adanorm"] = "cat"
    bidirectional: bool = True
    p_dropout: float = 0.1


class RNNEncoder(CNNEncoder):
    params: RNNEncoderParams

    def __init__(self, params: RNNEncoderParams, input_dim):
        super().__init__(params, input_dim)

        in_dim = super().output_dim

        if not params.condition:
            self.lstm = nn.LSTM(
                in_dim,
                params.encoder_inner_dim // (params.bidirectional + 1),
                num_layers=params.encoder_num_layers,
                batch_first=True,
                bidirectional=params.bidirectional,
                dropout=params.p_dropout,
            )
        elif params.condition_type == "cat":
            self.lstms = nn.ModuleList()
            for i in range(params.encoder_num_layers):
                if i == 0:
                    in_dim = in_dim + params.condition_dim
                else:
                    in_dim = params.encoder_inner_dim + params.condition_dim

                self.lstms.append(
                    nn.LSTM(
                        in_dim,
                        params.encoder_inner_dim // (params.bidirectional + 1),
                        num_layers=1,
                        batch_first=True,
                        bidirectional=params.bidirectional,
                        dropout=params.p_dropout,
                    )
                )
        elif params.condition_type == "adanorm":
            in_dim += params.condition_dim

            self.lstms = nn.ModuleList()
            for i in range(params.encoder_num_layers):
                if i == 0:
                    self.lstms.append(
                        nn.LSTM(
                            in_dim,
                            params.encoder_inner_dim // (params.bidirectional + 1),
                            num_layers=1,
                            batch_first=True,
                            bidirectional=params.bidirectional,
                            dropout=params.p_dropout,
                        )
                    )
                else:
                    self.lstms.append(
                        nn.LSTM(
                            params.encoder_inner_dim + params.condition_dim,
                            params.encoder_inner_dim // (params.bidirectional + 1),
                            num_layers=1,
                            batch_first=True,
                            bidirectional=params.bidirectional,
                            dropout=params.p_dropout,
                        )
                    )

                if i + 1 != params.encoder_num_layers:
                    self.lstms.append(
                        AdaLayerNorm(params.encoder_inner_dim, params.condition_dim)
                    )

        self.proj = Regression(params.encoder_inner_dim, self.params.encoder_output_dim)

    @property
    def output_dim(self):
        return self.params.encoder_output_dim

    def forward_step(self, inputs: ComponentInput) -> EncoderOutput:  # type: ignore
        inputs = super().forward_step(inputs)

        x, x_lens, x_mask = self.get_content_and_mask(inputs)

        if not self.params.condition:
            x = run_rnn_on_padded_sequence(self.lstm, x, x_lens)
        else:
            cond = self.get_condition(inputs, self.params.condition)
            cond = cond.squeeze(1)

            s = cond.unsqueeze(1).expand(x.shape[0], x.shape[1], -1)

            for block in self.lstms:
                if isinstance(block, AdaLayerNorm):
                    x = block(x, cond)
                    continue

                x = torch.cat([x, s], dim=2)
                x = apply_mask(x, x_mask)
                x = run_rnn_on_padded_sequence(block, x, x_lens)

        y = self.proj(x)

        outputs = EncoderOutput.copy_from(inputs)
        outputs = outputs.set_content(y).apply_mask(x_mask)
        outputs.encoder_context = x
        return outputs
