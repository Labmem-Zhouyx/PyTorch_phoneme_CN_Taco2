""" adapted from https://github.com/r9y9/tacotron_pytorch """
""" with reference to https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer """
""" adapted from https://github.com/NVIDIA/tacotron2 """

import torch
from torch import nn

from .grl import GradientReversal


class BatchNormConv1dStack(nn.Module):
    """
    BatchNormConv1dStack
        - A stack of 1-d convolution layers
        - Each convolution layer is followed by activation function (optional), Batch Normalization (BN) and dropout
    """
    def __init__(self, in_channel,
                 out_channels=[512, 512, 512], kernel_size=3, stride=1, padding=1,
                 activations=None, dropout=0.5):
        super(BatchNormConv1dStack, self).__init__()

        # Validation check
        if activations is None:
            activations = [None] * len(out_channels)
        assert len(activations) == len(out_channels)

        # 1-d convolutions with BN
        in_sizes = [in_channel] + out_channels[:-1]
        self.conv1ds = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=kernel_size, stride=stride,
                             padding=padding, activation=ac)
             for (in_size, out_size, ac) in zip(in_sizes, out_channels, activations)])

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for conv1d in self.conv1ds:
            x = self.dropout(conv1d(x))
        return x


class BatchNormConv1d(nn.Module):
    """
    BatchNormConv1d
        - 1-d convolution layer with specific activation function, followed by Batch Normalization (BN)
    Batch Norm before activation or after the activation?
    Still in debation!
    In practace, applying batch norm after the activation yields bettr results.
        - https://blog.paperspace.com/busting-the-myths-about-batch-normalization/
        - https://medium.com/@nihar.kanungo/batch-normalization-and-activation-function-sequence-confusion-4e075334b4cc
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class AdversarialClassifier(nn.Module):
    """
    AdversarialClassifier
        - 1 gradident reversal layer
        - n hidden linear layers with ReLU activation
        - 1 output linear layer with Softmax activation
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[256], rev_scale=1):
        """
        Args:
             in_dim: input dimension
            out_dim: number of units of output layer (number of classes)
        hidden_dims: number of units of hidden layers
          rev_scale: gradient reversal scale
        """
        super(AdversarialClassifier, self).__init__()

        self.gradient_rev = GradientReversal(rev_scale)

        in_sizes = [in_dim] + hidden_dims[:]
        out_sizes = hidden_dims[:] + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.activations = [nn.ReLU()] * len(hidden_dims) + [nn.Softmax(dim=-1)]

    def forward(self, x):
        x = self.gradient_rev(x)
        for (linear, activate) in zip(self.layers, self.activations):
            x = activate(linear(x))
        return x