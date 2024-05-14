import typing as tp

from copy import deepcopy
from dataclasses import dataclass

import torch

from torch import Tensor

from speechflow.training.utils.tensor_utils import apply_mask
from tts.acoustic_models.data_types import (
    TTSForwardInput,
    TTSForwardInputWithPrompt,
    TTSForwardInputWithSSML,
)

__all__ = [
    "ComponentInput",
    "ComponentOutput",
    "EncoderOutput",
    "VarianceAdaptorOutput",
    "DecoderOutput",
    "PostnetOutput",
]

MODEL_INPUT_TYPE = tp.Union[
    TTSForwardInput, TTSForwardInputWithPrompt, TTSForwardInputWithSSML
]


@dataclass
class ComponentInput:
    content: tp.Union[Tensor, tp.List[Tensor]]
    content_lengths: tp.Union[Tensor, tp.List[Tensor]]
    embeddings: tp.Dict[str, Tensor] = None  # type: ignore
    additional_content: tp.Dict[str, Tensor] = None  # type: ignore
    additional_losses: tp.Dict[str, Tensor] = None  # type: ignore
    model_inputs: MODEL_INPUT_TYPE = None  # type: ignore

    def __post_init__(self):
        if self.embeddings is None:
            self.embeddings = {}
        if self.additional_content is None:
            self.additional_content = {}
        if self.additional_losses is None:
            self.additional_losses = {}

    @property
    def device(self):
        if self.model_inputs is not None:
            return self.model_inputs.device
        elif self.content is not None:
            if isinstance(self.content, list):
                return self.content[0].device
            else:
                return self.content.device
        elif self.embeddings:
            return list(self.embeddings.values())[0].data.device
        else:
            raise RuntimeError("device not found")

    @staticmethod
    def empty():
        return ComponentInput(
            content=None,  # type: ignore
            content_lengths=None,  # type: ignore
        )

    @classmethod
    def copy_from(cls, x: "ComponentInput", deep: bool = False):
        new = cls(
            content=x.content,
            content_lengths=x.content_lengths,
            embeddings=x.embeddings,
            model_inputs=x.model_inputs,
            additional_content=x.additional_content,
            additional_losses=x.additional_losses,
        )
        return deepcopy(new) if deep else new

    def set_content(self, x: Tensor, x_lens: tp.Optional[Tensor] = None):
        self.content = x
        if x_lens is not None:
            self.content_lengths = x_lens
        return self

    def copy_content(self, detach: bool = False) -> tp.Union[Tensor, tp.List[Tensor]]:
        if isinstance(self.content, list):
            if detach:
                return [t.clone().detach() for t in self.content]
            else:
                return [t.clone() for t in self.content]
        else:
            if detach:
                return self.content.clone().detach()
            else:
                return self.content.clone()

    def cat_content(self, dim: int = 0):
        if isinstance(self.content, list):
            return torch.cat(self.content, dim=dim)
        else:
            return self.content

    def stack_content(self):
        if isinstance(self.content, list):
            return torch.stack(self.content)
        else:
            return self.content.unsqueeze(0)

    def apply_mask(self, mask: Tensor):
        self.content = apply_mask(self.content, mask)
        return self


ComponentOutput = ComponentInput


@dataclass
class EncoderOutput(ComponentOutput):
    encoder_context: Tensor = None


@dataclass
class VarianceAdaptorOutput(ComponentOutput):
    masks: tp.Dict[str, Tensor] = None
    attention_weights: Tensor = None
    variance_predictions: tp.Dict[str, Tensor] = None  # type: ignore


@dataclass
class DecoderOutput(ComponentOutput):
    decoder_context: Tensor = None
    gate: Tensor = None


@dataclass
class PostnetOutput(ComponentOutput):
    pass
