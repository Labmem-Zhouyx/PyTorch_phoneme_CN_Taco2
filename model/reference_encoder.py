import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_mask_from_lengths


class ReferenceEncoder(nn.Module):
    def __init__(self, hparams):
        super(ReferenceEncoder, self).__init__()

        in_dim = hparams.mel_dim
        conv_channels = hparams.ref_conv_channels
        gru_units = hparams.ref_gru_units
        K = len(conv_channels)
        kernel_size = 3
        stride = hparams.ref_conv_stride
        padding = 1
        self.convtimes = 1
        for i in stride:
            self.convtimes *= i


        # 2-D convolution layers
        filters = [1] + conv_channels
        self.conv2ds = nn.ModuleList(
            [nn.Conv2d(in_channels=filters[i],
                       out_channels=filters[i+1],
                       kernel_size=kernel_size,
                       stride=stride[i],
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
                          # bidirectional=True,
                          batch_first=True)

        self.global_outlayer = nn.Sequential(
            nn.Linear(in_features=gru_units,
                      out_features=hparams.speaker_embedding_dim),
            nn.Tanh()
        )

        self.local_outlayer = nn.Sequential(
            nn.Linear(in_features=gru_units,
                      out_features=hparams.ref_local_style_dim * 2),
            nn.Tanh()
        )


    def forward(self, inputs, input_lengths=None):
        out = inputs.unsqueeze(1)  # [B, 1, T, mel_dim]
        for conv, bn in zip(self.conv2ds, self.bns):
            out = conv(out)
            out = bn(out)
            out = self.relu(out)  # [B, 128, T//2^K, mel_dim//2^K], where 128 = conv_channels[-1]

        out = out.transpose(1, 2)  # [B, T//2^K, 128, mel_dim//2^K]
        B, T = out.size(0), out.size(1)
        out = out.contiguous().view(B, T, -1)  # [B, T//2^K, 128*mel_dim//2^K]

        self.gru.flatten_parameters()
        # get precise last step by excluding paddings
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / self.convtimes)
            input_lengths = input_lengths.cpu().numpy().astype(int)
            out = nn.utils.rnn.pack_padded_sequence(out, input_lengths, batch_first=True, enforce_sorted=False)
            memory, out = self.gru(out)  # memory --- [B, T, gru_units]  out --- [1, B, gru_units]
            memory, _ = nn.utils.rnn.pad_packed_sequence(memory, batch_first=True)
        else:
            memory, out = self.gru(out)  # memory --- [B, T, gru_units]  out --- [1, B, gru_units]

        global_embedding = self.global_outlayer(out.squeeze(0))
        local_embedding = self.local_outlayer(memory)

        return global_embedding.unsqueeze(1), local_embedding

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride[i] + 1
        return L


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

    def forward(self, text_embeddings, text_lengths, local_embeddings, ref_mels, ref_mel_lengths):

        key, value = torch.split(local_embeddings, self.ref_local_style_dim, dim=-1)  # [N, Ty, style_embedding_dim] * 2

        if text_lengths == None and ref_mel_lengths == None:
            attn_mask = None
        else:
            # Get attention mask
            # 1. text mask
            text_total_length = text_embeddings.size(1)  # [N, T_x, #dim]
            text_mask = get_mask_from_lengths(text_lengths, text_total_length).float().unsqueeze(-1)  # [B, seq_len, 1]
            # 2. mel mask (regularized to phoneme_scale)
            ref_mel_total_length = ref_mels.size(1)  # [N, T_y, n_mels]
            ref_mel_mask = get_mask_from_lengths(ref_mel_lengths, ref_mel_total_length).float().unsqueeze(-1)  # [B, rseq_len, 1]
            ref_mel_mask = F.interpolate(ref_mel_mask.transpose(1, 2), size=key.size(1))  # [B, 1, Ty]
            # 3. The attention mask
            attn_mask = torch.bmm(text_mask, ref_mel_mask)  # [N, seq_len, ref_len]

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
