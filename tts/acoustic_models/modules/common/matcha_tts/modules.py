from torch import nn

from tts.acoustic_models.modules.common.blocks import Regression

__all__ = ["AdaLayerNorm", "AdaLayerNormZero"]


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(self, time_emb_dim, cond_dim, embedding_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, embedding_dim)
        self.cond_proj = Regression(cond_dim, embedding_dim)

    def forward(self, timestep, cond):
        return self.time_proj(timestep) + self.cond_proj(cond).squeeze(1)


class AdaLayerNorm(nn.Module):
    """Norm layer adaptive layer norm."""

    def __init__(self, embedding_dim, time_emb_dim, cond_dim):
        super().__init__()
        self.emb = CombinedTimestepConditionEmbeddings(
            time_emb_dim, cond_dim, embedding_dim
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep, condition):
        emb = self.linear(self.silu(self.emb(timestep, condition)))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class AdaLayerNormZero(nn.Module):
    """Norm layer adaptive layer norm zero (adaLN-Zero)."""

    def __init__(self, embedding_dim, time_emb_dim, cond_dim):
        super().__init__()
        self.emb = CombinedTimestepConditionEmbeddings(
            time_emb_dim, cond_dim, embedding_dim
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, condition):
        emb = self.linear(self.silu(self.emb(timestep, condition)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
