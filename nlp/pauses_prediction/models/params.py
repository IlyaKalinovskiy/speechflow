from dataclasses import dataclass

from tts.acoustic_models.modules.params import EmbeddingParams

__all__ = ["PausesPredictionParams"]


@dataclass
class PausesPredictionParams(EmbeddingParams):
    encoder_emb_dim: int = 64
    encoder_rnn_dim: int = 64
    use_convolutions: bool = True
    encoder_kernel_size: int = 3
    encoder_n_convolutions: int = 2
    encoder_num_additional_seqs: int = 4
    dropout: float = 0.5
    decoder_dim: int = 100
    token_emb_dim: int = 64
