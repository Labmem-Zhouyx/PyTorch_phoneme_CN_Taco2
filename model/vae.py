import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, mel_dim, latent_dim=16, conv_channels=[512, 512], lstm_units=256, lstm_layers=2, kernel_size=3,
                 stride=2, padding=1):
        super().__init__()
        self.encoder = ReferenceEncoder(mel_dim, conv_channels, lstm_units, lstm_layers, kernel_size, stride, padding)
        self._latent_mean = nn.Linear(in_features=lstm_units * 2, out_features=latent_dim)
        self._latent_mean = nn.Linear(in_features=lstm_units * 2, out_features=latent_dim)
        self._latent_var = nn.Linear(in_features=lstm_units * 2, out_features=latent_dim)

    def forward(self, inputs, input_lengths=None):
        enc_outs = self.encoder(inputs, input_lengths)
        enc_out = torch.mean(enc_outs, axis=1)
        latent_mean = self._latent_mean(enc_out)
        latent_var = self._latent_var(enc_out)
        out = self.reparameterize(latent_mean, latent_var)
        return out, latent_mean, latent_var

    def reparameterize(self, mu, log_var):
        "Reparameterize from mean and variance"
        device = next(self.parameters()).device
        eps = torch.randn(log_var.shape).to(device).float()
        std = torch.exp(log_var) ** 0.5
        z = mu + std * eps
        return z


class ReferenceEncoder(nn.Module):
    """
    ReferenceEncoder
        - a mel spectrogram is first passed through two convolutional layers, which contains 512 filters with shape 3 × 1.
        - The output of these convolutional layers is then fed to a stack of two bidirectional LSTM layers with 256 cells at each direction.
        - A mean pooling layer is used to summarize the LSTM outputs across time, followed by a linear projection layer to predict the
        posterior mean and log variance.
    """

    def __init__(self, in_dim, conv_channels=[512, 512], lstm_units=256, lstm_layers=2, kernel_size=3, stride=2, padding=1):
        super().__init__()

        K = len(conv_channels)

        # 2-D convolution layers
        filters = [1] + conv_channels
        self.conv2ds = nn.ModuleList(
            [nn.Conv2d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding)
             for i in range(K)])

        # 2-D batch normalization (BN) layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=conv_channels[i])
             for i in range(K)])

        # ReLU
        self.relu = nn.ReLU()

        # LSTM
        out_channels = self.calculate_channels(in_dim, kernel_size, stride, padding, K)
        self.blstms = nn.LSTM(input_size=conv_channels[-1] * out_channels,
                              hidden_size=lstm_units,
                              num_layers=lstm_layers,
                              batch_first=True,
                              bidirectional=True)



    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim] mels
            input_lengths --- [B] lengths of the mels
        output:
            out --- [B, lstm_units * 2]
        """

        out = inputs.unsqueeze(1)  # [B, 1, T, mel_dim]
        for conv, bn in zip(self.conv2ds, self.bns):
            out = conv(out)
            out = bn(out)
            out = self.relu(out)   # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        # get precise last step by excluding paddings
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.conv2ds))
            input_lengths = input_lengths.cpu().numpy().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(out, input_lengths, batch_first=True, enforce_sorted=False)

        self.blstms.flatten_parameters()
        out, _ = self.blstms(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)   # out --- [B, T, 2 * lstm_units]

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

