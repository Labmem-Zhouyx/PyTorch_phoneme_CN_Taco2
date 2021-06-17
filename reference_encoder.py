import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_mask_from_lengths


class ReferenceEncoder(nn.Module):
    def __init__(self, hparams):
        super(ReferenceEncoder, self).__init__()

        K = len(hparams.ref_conv_channels)
        filters = [hparams.mel_dim] + hparams.ref_conv_channels
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=filters[i],
                            out_channels=filters[i + 1],
                            kernel_size=3,
                            stride=1,
                            padding=1) for i in range(K)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=hparams.ref_conv_channels[i]) for i in range(K)])

        self.global_bgru = nn.GRU(input_size=hparams.ref_conv_channels[-1],
                            hidden_size=hparams.ref_global_gru_units,
                            bidirectional=True,
                            batch_first=True)

        self.global_outlayer = nn.Sequential(
            nn.Linear(in_features=hparams.ref_global_gru_units * 2,
                      out_features=hparams.speaker_embedding_dim),
            nn.Tanh()
        )

        self.local_gru = nn.GRU(input_size=hparams.ref_conv_channels[-1],
                           hidden_size=hparams.ref_local_gru_units,
                           batch_first=True)

        self.local_outlayer = nn.Sequential(
            nn.Linear(in_features=hparams.ref_local_gru_units,
                      out_features=hparams.ref_local_style_dim * 2),
            nn.Tanh()
        )

    def forward(self, inputs):
        x = inputs.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
        conv_out = x.transpose(1, 2)
        bmemory, _ = self.global_bgru(conv_out)
        global_embedding = self.global_outlayer(bmemory[:, -1])

        memory, _ = self.local_gru(conv_out)
        local_embedding = self.local_outlayer(memory)

        return global_embedding, local_embedding


class ScaledDotProductAttention(
    nn.Module):  # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    ''' Scaled Dot-Product Attention '''

    def __init__(self, hparams):
        super().__init__()
        self.dropout = nn.Dropout(hparams.ref_attention_dropout) \
            if hparams.ref_attention_dropout > 0 else None

        self.d_q = hparams.encoder_blstm_units
        self.d_k = hparams.ref_local_style_dim

        self.linears = nn.ModuleList([
            LinearNorm(in_dim, hparams.ref_attention_dim, bias=False, w_init_gain='tanh') \
            for in_dim in (self.d_q, self.d_k)
        ])

        self.score_mask_value = -1e9

    def forward(self, q, k, v, mask=None):
        q, k = [linear(vector) for linear, vector in zip(self.linears, (q, k))]

        alignment = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [N, seq_len, ref_len]

        if mask is not None:
            alignment = alignment.masked_fill_(mask == 0, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=-1)
        attention_weights = self.dropout(attention_weights) \
            if self.dropout is not None else attention_weights

        attention_context = torch.bmm(attention_weights, v)  # [N, seq_len, prosody_embedding_dim]

        return attention_context, attention_weights


class LocalAttentionEncoder(nn.Module):
    '''
    embedded_text --- [N, seq_len, text_embedding_dim]
    mels --- [N, n_mels*r, Ty/r], r=1
    style_embed --- [N, seq_len, style_embedding_dim]
    alignments --- [N, seq_len, ref_len], Ty/r = ref_len
    '''
    def __init__(self, hparams):
        super(LocalAttentionEncoder, self).__init__()
        self.ref_local_style_dim = hparams.ref_local_style_dim
        self.ref_attn = ScaledDotProductAttention(hparams)

    def forward(self, text_embeddings, text_lengths, local_embeddings, mels, mels_lengths):

        key, value = torch.split(local_embeddings, self.ref_local_style_dim, dim=-1)  # [N, Ty, style_embedding_dim] * 2

        if text_lengths == None and mels_lengths == None:
            attn_mask = None
        else:
            # Get attention mask
            # 1. text mask
            text_total_length = text_embeddings.size(1)  # [N, T_x, #dim]
            text_mask = get_mask_from_lengths(text_lengths, text_total_length).float().unsqueeze(-1)  # [B, seq_len, 1]
            # 2. mel mask (regularized to phoneme_scale)
            mels_total_length = mels.size(2)  # [N, #n_mels, T_y]
            mels_mask = get_mask_from_lengths(mels_lengths, mels_total_length).float().unsqueeze(-1)  # [B, rseq_len, 1]
            mels_mask = F.interpolate(mels_mask.transpose(1, 2), size=key.size(1))  # [B, 1, Ty]
            # 3. The attention mask
            attn_mask = torch.bmm(text_mask, mels_mask)  # [N, seq_len, ref_len]

        # Attention
        style_embed, alignments = self.ref_attn(text_embeddings, key, value, attn_mask)

        # Apply ReLU as the activation function to force the values of the prosody embedding to lie in [0, ∞].
        style_embed = F.relu(style_embed)

        return style_embed, alignments


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
