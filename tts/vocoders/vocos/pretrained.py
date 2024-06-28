from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import yaml
import torch

from huggingface_hub import hf_hub_download
from torch import nn

from tts.vocoders.data_types import VocoderInferenceInput, VocoderInferenceOutput
from tts.vocoders.vocos.modules.backbone import Backbone
from tts.vocoders.vocos.modules.feature_extractors import FeatureExtractor
from tts.vocoders.vocos.modules.heads import FourierHead


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.

    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    """The Vocos class represents a Fourier-based neural vocoder for audio synthesis.

    This class is primarily designed for inference, with support for loading from
    pretrained model checkpoints. It consists of three main components: a feature
    extractor, a backbone, and a head.

    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str, config: Dict = None) -> Vocos:
        """Class method to create a new Vocos model instance from hyperparameters stored
        in a yaml configuration file."""
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: Optional[str] = None) -> Vocos:
        """Class method to create a new Vocos model instance from a pre-trained model
        stored in the Hugging Face model hub."""
        config_path = hf_hub_download(
            repo_id=repo_id, filename="config.yaml", revision=revision
        )
        model_path = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", revision=revision
        )
        model = cls.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Method to run a copy-synthesis from audio waveform. The feature extractor first
        processes the audio input, which is then passed through the backbone and the head
        to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).

        """
        features, _ = self.feature_extractor(audio_input, **kwargs)
        audio_output = self.decode(features, **kwargs)
        return audio_output

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Method to decode audio waveform from already calculated features. The features
        input is passed through the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).

        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def inference(
        self, inputs: VocoderInferenceInput, **kwargs
    ) -> VocoderInferenceOutput:
        feat, additional_content = self.feature_extractor(inputs, **kwargs)
        waveform, _, _ = self.decode(feat, **kwargs)
        return VocoderInferenceOutput(
            waveform=waveform, additional_content=additional_content
        )
