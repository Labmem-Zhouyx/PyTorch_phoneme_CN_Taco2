""" adapted from https://github.com/r9y9/tacotron_pytorch """
""" with reference to https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer """
""" adapted from https://github.com/NVIDIA/tacotron2 """

import torch
from torch import nn


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