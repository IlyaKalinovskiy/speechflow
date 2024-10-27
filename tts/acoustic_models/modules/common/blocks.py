import math
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from speechflow.utils.tensor_utils import run_rnn_on_padded_sequence
from tts.acoustic_models.modules.common.layers import Conv, HighwayNetwork

__all__ = ["VarianceEmbedding", "Regression", "ConvPrenet", "CBHG"]


class VarianceEmbedding(nn.Module):
    def __init__(
        self,
        interval: tp.Tuple[float, ...],
        n_bins: int = 256,
        emb_dim: int = 32,
        log_scale: bool = False,
        activation_fn: str = "Tanh",
    ):
        super().__init__()
        v_min, v_max = interval
        self.log_scale = log_scale

        if self.log_scale:
            v_min, v_max = np.log1p(v_min), np.log1p(v_max)

        self.bins = torch.linspace(v_min, v_max, n_bins - 1)

        self.embedding = nn.Embedding(n_bins, emb_dim)
        self.af = getattr(nn, activation_fn)()

    def forward(self, x: torch.Tensor):
        embedding = None

        if self.bins.device != x.device:
            self.bins = self.bins.to(x.device)

        if x.ndim == 2:
            y = torch.log1p(x) if self.log_scale else x
            embedding = self.af(self.embedding(torch.bucketize(y, self.bins)))

        elif x.ndim == 3:
            temp = []
            for i in range(x.shape[2]):
                y = torch.log1p(x[:, :, i]) if self.log_scale else x[:, :, i]
                temp.append(self.af(self.embedding(torch.bucketize(y, self.bins))))

            embedding = torch.cat(temp, dim=2)

        return embedding.squeeze(1)


class Regression(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        p_dropout: float = 0.1,
        activation_fn: str = "Identity",
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.af1 = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.af2 = getattr(nn, activation_fn)()

    def forward(self, x, x_mask=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        y = self.af1(self.linear1(x).transpose(2, 1))
        y = self.af2(self.linear2(self.dropout(y.transpose(2, 1))))
        return y


class ConvPrenet(nn.Module):
    """Default Prenet Block.

    Consist of Reflection padding and Convolutional layer.
    x -> ReflectionPadding -> Conv1D -> Activation -> y

    Args:
        in_channels (int): Input channels to Encoder Block.
        out_channels (int): Output channels from Encoder Block.
        kernel_size (int): Kernel size for Conv1D. Defaults to 3.

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd integer, got kernel_size={kernel_size}"
            )

        _padding_size = math.ceil((kernel_size - 1) / 2)

        self.padding = nn.ReflectionPad1d(_padding_size)
        self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.activation = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, Length)

        Returns:
            y (torch.Tensor): Output tensor of shape (B, out_channels, Length)
        """
        x = self.padding(x)
        x = self.encoder(x)
        x = self.activation(x)
        return x


class CBHG(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_banks_num: int,
        highways_num: int,
        kernel_size: int = 3,
        rnn_dim: tp.Optional[int] = None,
        rnn_bidirectional: bool = True,
    ):
        super().__init__()
        if rnn_dim is None:
            rnn_dim = in_dim

        # List of all rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.bank_kernels = [kernel_size * (i + 1) for i in range(conv_banks_num)]
        self.conv1d_bank = nn.ModuleList()
        for k in self.bank_kernels:
            padding = math.ceil((k - 1) / 2)
            conv = Conv(
                in_dim,
                out_dim,
                kernel_size=k,
                padding=padding,
                bias=False,
                batch_norm=True,
                use_activation=True,
            )
            self.conv1d_bank.append(conv)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

        self.conv_project1 = Conv(
            conv_banks_num * out_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=math.ceil((kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )
        self.conv_project2 = Conv(
            out_dim,
            in_dim,
            kernel_size=kernel_size,
            padding=math.ceil((kernel_size - 1) / 2),
            bias=False,
            batch_norm=True,
            use_activation=True,
        )

        self.highways = nn.ModuleList()
        for i in range(highways_num):
            hn = HighwayNetwork(in_dim)
            self.highways.append(hn)

        self.rnn = nn.GRU(
            rnn_dim,
            out_dim // (2 if rnn_bidirectional else 1),
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )
        self._to_flatten.append(self.rnn)

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

        self.receptive_field = (torch.tensor(self.bank_kernels) - 1).sum() + 4 + 1 + 1

    def calc_rnn(self, x, seq_lengths=None):
        return run_rnn_on_padded_sequence(self.rnn, x, seq_lengths)

    def forward(self, x, seq_lengths=None):
        x = x.transpose(2, 1)

        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        # Save these for later
        residual = x
        conv_bank = []

        # Convolution Bank
        for idx, conv in enumerate(self.conv1d_bank):
            c = conv(x)
            if idx % 2 == 0:
                c = F.pad(c, [0, 1])
            conv_bank.append(c)

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        x = self.maxpool(conv_bank)

        # Conv1d projections
        x = self.conv_project1(x)
        x = self.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        for h in self.highways:
            x = h(x)

        # And then the RNN
        x = self.calc_rnn(x, seq_lengths)

        return x

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN.

        Used to improve efficiency and avoid PyTorch yelling at us.

        """
        [m.flatten_parameters() for m in self._to_flatten]
