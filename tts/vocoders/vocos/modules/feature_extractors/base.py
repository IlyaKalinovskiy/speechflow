import torch

from torch import nn

from tts.vocoders.data_types import VocoderForwardInput

__all__ = ["FeatureExtractor"]


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, inputs: VocoderForwardInput, **kwargs) -> torch.Tensor:
        """Extract features from the given audio.

        Args:
            inputs (VocoderForwardInput): Input audio features.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.

        """
        raise NotImplementedError("Subclasses must implement the forward method.")
