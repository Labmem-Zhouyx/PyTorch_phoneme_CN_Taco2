"""
Global Style Token
Code from:
https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
""" 
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class GST(nn.Module):
    """
    GlobalStyleToken (GST)
    GST is described in:
        Y. Wang, D. Stanton, Y. Zhang, R.J. Shkerry-Ryan, E. Battenberg, J. Shor, Y. Xiao, F. Ren, Y. Jia, R.A. Saurous,
        "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis,"
        in Proceedings of the 35th International Conference on Machine Learning (PMLR), 80:5180-5189, 2018.
        https://arxiv.org/abs/1803.09017
    See:
        https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
        https://github.com/NVIDIA/mellotron/blob/master/modules.py
    """

    def __init__(self, hparams):
        super().__init__()
        self.stl = STL(hparams)

    def forward(self, inputs):
        style_embed = self.stl(inputs)

        return style_embed


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, hparams):

        super().__init__()
        E = hparams.gst_num_units
        self.embed = nn.Parameter(torch.FloatTensor(hparams.gst_token_num, E // hparams.gst_head_num))
        d_q = E // 2
        d_k = E // hparams.gst_head_num
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=E, num_heads=hparams.gst_head_num)
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
