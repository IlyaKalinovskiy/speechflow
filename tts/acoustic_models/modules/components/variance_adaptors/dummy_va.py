import typing as tp

import torch

from speechflow.training.utils.tensor_utils import get_mask_from_lengths
from tts.acoustic_models.modules.component import Component
from tts.acoustic_models.modules.data_types import EncoderOutput, VarianceAdaptorOutput
from tts.acoustic_models.modules.params import VarianceAdaptorParams

__all__ = ["DummyVarianceAdaptor", "DummyVarianceAdaptorParams"]


class DummyVarianceAdaptorParams(VarianceAdaptorParams):
    pass


class DummyVarianceAdaptor(Component):
    params: DummyVarianceAdaptorParams

    def __init__(
        self,
        params: DummyVarianceAdaptorParams,
        input_dim: tp.Union[int, tp.Tuple[int, ...]],
    ):
        super().__init__(params, input_dim)

    @property
    def output_dim(self):
        if isinstance(self.input_dim, int):
            return (self.input_dim,)
        else:
            return tuple(self.input_dim)

    def forward_step(self, inputs: EncoderOutput) -> VarianceAdaptorOutput:  # type: ignore
        model_inputs = inputs.model_inputs
        src_lengths = model_inputs.input_lengths
        text_lengths = model_inputs.text_lengths
        spec_lengths = model_inputs.output_lengths
        max_spec_len = torch.max(spec_lengths)

        src_mask = get_mask_from_lengths(src_lengths, torch.max(src_lengths))
        text_mask = get_mask_from_lengths(text_lengths, max_spec_len)
        spec_mask = get_mask_from_lengths(spec_lengths, max_spec_len)

        masks = {
            "src": src_mask,
            "text": text_mask,
            "spec": spec_mask,
        }

        outputs = VarianceAdaptorOutput.copy_from(inputs)
        outputs.masks = masks
        outputs.variance_predictions = {}
        return outputs
