from dataclasses import dataclass

from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["ProsodyPredictionParams"]


@dataclass
class ProsodyPredictionParams(EmbeddingParams):
    model_name: str = None  # type: ignore
    dropout: float = 0.5
    n_classes: int = 10
    n_layers_tune: int = None  # type: ignore
    classification_task: str = "both"
