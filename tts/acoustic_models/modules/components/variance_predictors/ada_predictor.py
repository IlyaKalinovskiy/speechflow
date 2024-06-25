import typing as tp

import torch

from torch import nn

from tts.acoustic_models.modules.ada_speech.decoder import AdaDecoder, AdaDecoderParams
from tts.acoustic_models.modules.component import MODEL_INPUT_TYPE, Component
from tts.acoustic_models.modules.data_types import ComponentOutput
from tts.acoustic_models.modules.params import VariancePredictorParams

__all__ = ["AdaPredictor", "AdaPredictorParams"]


class AdaPredictorParams(VariancePredictorParams):
    aggregate_by_tokens: bool = False
    condition: tp.Tuple[str, ...] = ()
    condition_dim: int = 0


class AdaPredictor(Component):
    params: AdaPredictorParams

    def __init__(
        self, params: AdaPredictorParams, input_dim: tp.Union[int, tp.Tuple[int, ...]]
    ):
        super().__init__(params, input_dim)
        decoder_params = AdaDecoderParams(
            decoder_hidden_dim=params.vp_inner_dim,
            decoder_num_layers=params.vp_num_layers,
            decoder_output_dim=params.vp_latent_dim,
            max_output_length=params.max_input_length
            if params.aggregate_by_tokens
            else params.max_output_length,
            condition=params.condition,
            condition_dim=params.condition_dim,
        )
        self.decoder = AdaDecoder(decoder_params, input_dim)
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.decoder.output_dim, self.decoder.output_dim),
            nn.ReLU(),
            nn.Linear(self.decoder.output_dim, params.vp_output_dim),
        )

    @property
    def output_dim(self):
        return self.params.vp_output_dim

    def forward_step(
        self, x, x_lengths, model_inputs: MODEL_INPUT_TYPE, **kwargs
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]]:
        dec_input = ComponentOutput.empty()
        dec_input.content = x
        dec_input.content_lengths = x_lengths
        dec_input.model_inputs = model_inputs
        for name in self.params.condition:
            dec_input.additional_content[name] = model_inputs.additional_inputs[name]

        if dec_input.content_lengths is None:
            dec_input.content_lengths = x_lengths

        det_outputs = self.decoder(dec_input)

        x = self.predictor(det_outputs.content).squeeze(-1)
        return x, {}, {}
