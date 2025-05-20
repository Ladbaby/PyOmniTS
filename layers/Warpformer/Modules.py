import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(-2, -1))
        
        if mask is not None and mask.dim() == 5:
            mask = mask.transpose(2, 4)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.bmm(attn, v)

        return output, attn


class ScaledDotProductAttention_bias(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, temperature, attn_dropout=0.2):
        super().__init__()
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head

    def forward(self, q, k, v, mask):
        # [B,K,H,LQ,LK] for temporal, [B,L,H,Kq,Kk] for category
        # [B,K,L,H,D]
        # attn = rearrange(self.w_qs(q), 'b k l (n d) -> b k n l d', n=self.n_head)
        q = rearrange(self.w_qs(q), 'b k l (n d) -> b k n l d', n=self.n_head)
        k = rearrange(self.w_ks(k), 'b k l (n d) -> b k n d l', n=self.n_head)
        v = rearrange(self.w_vs(v), 'b k l (n d) -> b k n l d', n=self.n_head)
        # v = rearrange(v, 'b k l d -> b k 1 l d')
        
        attn = torch.matmul(q , k) / self.temperature

        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(2).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        v = torch.matmul(attn, v)

        v = rearrange(v, 'b k n l d -> b k l (n d)')

        return v, attn
    
class Attention(nn.Module):

    def __init__(self, hin_d, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, hin_d)
        self.W = nn.Linear(hin_d,1, bias=False)
        
    def forward(self, x, mask=None, mask_value=-1e30):
        # [B,K,L,D]
        
        # map directly
        attn = self.W(torch.tanh(self.linear(x))) # [B,K,L,1]

        if mask is not None:
            attn = mask * attn + (1-mask)*mask_value
            # attn = attn.masked_fill(mask, mask_value)
            
        attn = F.softmax(attn, dim=-2)
        
        x = torch.matmul(x.transpose(-1, -2), attn).squeeze(-1) # [B,K,D,1]

        return x, attn