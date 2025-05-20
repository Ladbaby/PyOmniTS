import torch.nn as nn
import torch
from .SubLayers import  PositionwiseFeedForward, MultiHeadAttention_tem_bias, MultiHeadAttention_type_bias
from einops import rearrange, repeat

PAD = 0

def get_attn_key_pad_mask_K(mask, transpose=False, full_attn=False):
    """ For masking out the padding part of key sequence. """
    if full_attn:
        if transpose:
            mask = rearrange(mask, 'b l k -> b k l')
        padding_mask = repeat(mask, 'b k l1 -> b k l2 l1', l2=mask.shape[-1]).eq(PAD)
    else:
        if transpose:
            seq_q = rearrange(mask, 'b l k -> b k l 1')
            seq_k = rearrange(mask, 'b l k -> b k 1 l')
        else:
            seq_q = rearrange(mask, 'b k l -> b k l 1')
            seq_k = rearrange(mask, 'b k l -> b k 1 l')
        padding_mask = torch.matmul(seq_q, seq_k).eq(PAD)

    return padding_mask

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, configs=None):
        super(EncoderLayer, self).__init__()

        self.full_attn = False

        self.slf_tem_attn = MultiHeadAttention_tem_bias(
            n_head, d_model, d_k, d_v, dropout=dropout, configs=configs)

        self.slf_type_attn = MultiHeadAttention_type_bias(
            n_head, d_model, d_k, d_v, dropout=dropout)


        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input, non_pad_mask=None):
        # time attention
        # [B, K, L, D]
        tem_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=False, full_attn=self.full_attn)
        type_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=True, full_attn=self.full_attn)
        
        # residue = enc_input
        tem_output = self.layer_norm(input)
        
        tem_output, enc_tem_attn = self.slf_tem_attn(
            tem_output, tem_output, tem_output, mask=tem_mask) 
        
        tem_output = tem_output + input

        tem_output = rearrange(tem_output, 'b k l d -> b l k d')

        # type attention
        # [B, L, K, D]
        # residue = enc_output
        type_output = self.layer_norm(tem_output)
        
        type_output, enc_type_attn = self.slf_type_attn(
            type_output, type_output, type_output, mask=type_mask) 
        
        enc_output = type_output + tem_output
        
        # FFFNN
        # residue = enc_output
        output = self.layer_norm(enc_output)
        
        output = self.pos_ffn(output)

        output = output + enc_output
        
        output = rearrange(output, 'b l k d -> b k l d')
        
        # optional
        output = self.layer_norm(output)

        return output, enc_tem_attn, enc_type_attn


