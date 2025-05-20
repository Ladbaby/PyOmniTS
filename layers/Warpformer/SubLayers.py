import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention_bias

class MultiHeadAttention_tem_bias(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, configs=None):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.configs = configs

        self.fc = nn.Linear(d_v * n_head, d_model)

        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # event_matrix [B,L,K]

        # [B,K,H,Lq,Lk]
        output, attn = self.attention(q, k, v, mask=mask) # [B,K,H,L,D]

        output = self.dropout(self.fc(output))

        return output, attn


class MultiHeadAttention_type_bias(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # [B,L,K,D]
        output, attn = self.attention(q, k, v, mask=mask) 

        output = self.dropout(self.fc(output))

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


