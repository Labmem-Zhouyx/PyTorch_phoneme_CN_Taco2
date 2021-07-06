import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, mel_dim, latent_dim=16, out_dim=256, gru_units=128, conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3,
                 stride=2, padding=1):
        super().__init__()
        self.encoder = ReferenceEncoder(mel_dim, gru_units, conv_channels, kernel_size, stride, padding)
        self._latent_mean = nn.Linear(in_features=gru_units, out_features=latent_dim)
        self._latent_var = nn.Linear(in_features=gru_units, out_features=latent_dim)
        self.latent_dim = latent_dim
        self.output = nn.Linear(in_features=latent_dim, out_features=out_dim)

    def forward(self, inputs, input_lengths=None, is_sampling=True):
        if inputs == None: # inference without reference
            out = torch.FloatTensor(1, 1, self.latent_dim).to('cuda')
            out.zero_()
            out = self.output(out)
            return out, None, None
        else:
            enc_out = self.encoder(inputs, input_lengths).unsqueeze(1)
            latent_mean = self._latent_mean(enc_out)
            latent_logvar = self._latent_var(enc_out)
            if is_sampling:
                out = self.reparameterize(latent_mean, latent_logvar)
            else:
                out = latent_mean
            out = self.output(out)
            return out, latent_mean, latent_logvar

    def reparameterize(self, mu, log_var):
        "Reparameterize from mean and variance"
        device = next(self.parameters()).device
        eps = torch.randn_like(log_var).to(device).float()
        std = torch.exp(0.5 * log_var)
        z = mu + std * eps
        return z


class ReferenceEncoder(nn.Module):
    """
    ReferenceEncoder
        - 6 2-D convolutional layers with 3*3 kernel, 2*2 stride, batch norm (BN), ReLU
        - a single-layer unidirectional GRU with 128-unit
    """

    def __init__(self, in_dim, gru_units=128, conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3, stride=2, padding=1):
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

        # GRU
        out_channels = self.calculate_channels(in_dim, kernel_size, stride, padding, K)
        self.gru = nn.GRU(input_size=conv_channels[-1] * out_channels,
                          hidden_size=gru_units,
                          batch_first=True)

    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim] mels
            input_lengths --- [B] lengths of the mels
        output:
            out --- [B, gru_units]
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

        self.gru.flatten_parameters()
        _, out = self.gru(out)  # out --- [1, B, gru_units]

        return out.squeeze(0)  # [B, gru_units]

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
