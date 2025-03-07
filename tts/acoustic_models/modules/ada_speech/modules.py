import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from speechflow.utils.tensor_utils import apply_mask, masked_fill
from tts.acoustic_models.modules.common.conditional_layers import ConditionalLayer


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = masked_fill(attn, mask, -torch.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, c_dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attn = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        if c_dim:
            self.layer_norm = ConditionalLayer("AdaNorm", d_model, c_dim)
        else:
            self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, c, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn_mask = self.attn(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        if isinstance(self.layer_norm, ConditionalLayer):
            output = self.layer_norm(output + residual, c)
        else:
            output = self.layer_norm(output + residual)

        return output, attn_mask


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, kernel_size, c_dim, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        if c_dim:
            self.layer_norm = ConditionalLayer("AdaNorm", d_in, c_dim)
        else:
            self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)

        if isinstance(self.layer_norm, ConditionalLayer):
            output = self.layer_norm(output + residual, c)
        else:
            output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block."""

    def __init__(
        self, d_model, n_head, d_k, d_v, d_inner, kernel_size, c_dim, dropout=0.1
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, c_dim, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, c_dim, dropout=dropout
        )

    def forward(self, enc_input, cond_emb, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, cond_emb, mask=slf_attn_mask
        )
        enc_output = apply_mask(enc_output, mask.unsqueeze(-1))

        enc_output = self.pos_ffn(enc_output, cond_emb)
        enc_output = apply_mask(enc_output, mask)

        return enc_output, enc_slf_attn
