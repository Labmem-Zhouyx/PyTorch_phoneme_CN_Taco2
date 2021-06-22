import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self._latent_mean = nn.Linear(in_features=in_dim, out_features=latent_dim)
        self._latent_var = nn.Linear(in_features=in_dim, out_features=latent_dim)

    def forward(self, inputs):
        latent_mean = self._latent_mean(inputs)
        latent_var = self._latent_var(inputs)
        out = self.reparameterize(latent_mean, latent_var)
        return out, latent_mean, latent_var

    def reparameterize(self, mu, log_var):
        "Reparameterize from mean and variance"
        device = next(self.parameters()).device
        eps = torch.randn(log_var.shape).to(device).float()
        std = torch.exp(log_var) ** 0.5
        z = mu + std * eps
        return z

