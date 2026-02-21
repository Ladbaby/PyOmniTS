'''
essentially the same as layers/GraFITi/GraFITi_layers.py
'''

import math

import torch
import torch.nn.functional as F
from torch import nn


class MAB2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super(MAB2, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim =n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K = torch.cat(K.split(dim_split, 2), 0)
        V = torch.cat(V.split(dim_split, 2), 0)

        Att_mat = Q_.bmm(K.transpose(1,2))/math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(mask.repeat(self.num_heads,1,1) == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class Encoder(nn.Module):
    def __init__(self, dim=41, nkernel=128, n_patches=8, n_layers=3, attn_head=4):
        super(Encoder, self).__init__()
        self.dim = dim + 2
        self.nheads = attn_head
        self.nkernel = nkernel
        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)
        self.n_layers = n_layers
        self.channel_time_attn = nn.ModuleList()
        self.time_channel_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        self.channel_attn = nn.ModuleList()
        self.output = nn.Linear(3 * nkernel, 1)
        self.register_parameter('corr', nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(n_patches, 1, 1)))
        for i in range(self.n_layers):
            self.channel_time_attn.append(MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.time_channel_attn.append(MAB2(nkernel, 2 * nkernel, 2 * nkernel, nkernel, self.nheads))
            self.edge_nn.append(nn.Linear(3 * nkernel, nkernel))
            self.channel_attn.append(MAB2(nkernel, nkernel, nkernel, nkernel, self.nheads))
        self.relu = nn.ReLU()
        self.score_layer = nn.Linear(nkernel, nkernel)

    def gather(self, x, inds):
        # inds =  # keep repeating until the embedding len as a new dim
        return x.gather(1, inds[:, :, None].repeat(1, 1, x.shape[-1]))

    def _normal_corr(self, n_batch, i_patch, eps=1e-6):
        corr = ((self.corr[i_patch] + self.corr[i_patch].T) / 2)
        corr_min, corr_max = torch.min(corr), torch.max(corr)
        corr = (corr - corr_min) / (corr_max - corr_min + eps)
        return corr.unsqueeze(0).repeat(n_batch, 1, 1)

    def forward(self, context_x, value, mask, target_mask, i_patch=0, x_repr_var=None):
        ndims = value.shape[-1]  # C
        T = context_x[:, :, None]  # BxTx1
        C = torch.ones([context_x.shape[0], ndims]).cumsum(1).to(value.device) - 1  # BxC intialization for one hot encoding channels
        T_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), 1) - 1  # BxTxC init for time indices
        C_inds = torch.cumsum(torch.ones_like(value).to(torch.int64), -1) - 1  # BxTxC init for channel indices
        mk_bool = mask.to(torch.bool)  # BxTxC
        full_len = torch.max(mask.sum((1, 2))).to(torch.int64)  # flattened TxC max length possible
        pad = lambda v: F.pad(v, [0, full_len - len(v)], value=0)

        # flattening to 2D
        T_inds_ = torch.stack([pad(r[m]) for r, m in zip(T_inds, mk_bool)]).contiguous()  # BxTxC -> Bxfull_len
        U_ = torch.stack([pad(r[m]) for r, m in zip(value, mk_bool)]).contiguous()  # BxTxC (values) -> Bxfull_len
        target_mask_ = torch.stack([pad(r[m]) for r, m in zip(target_mask, mk_bool)]).contiguous()  # BxK_
        C_inds_ = torch.stack([pad(r[m]) for r, m in zip(C_inds, mk_bool)]).contiguous()  # BxK_
        mk_ = torch.stack([pad(r[m]) for r, m in zip(mask, mk_bool)]).contiguous()  # BxK_
        source_, source_mask_ = U_.clone(), mk_ - target_mask_

        obs_len = full_len

        # C_ = torch.nn.functional.one_hot(C.to(torch.int64), num_classes=ndims).to(torch.float32)  # BxCxC #channel one hot encoding
        C_ = self._normal_corr(context_x.size(0), 0)
        U_indicator = 1 - mk_ + target_mask_
        U_ = torch.cat([U_[:, :, None], U_indicator[:, :, None]], -1)  # BxK_max x 2 #todo: correct

        # creating Channel mask and Time mask
        C_mask = C[:, :, None].repeat(1, 1, obs_len)
        temp_c_inds = C_inds_[:, None, :].repeat(1, ndims, 1)
        C_mask = (C_mask == temp_c_inds).to(torch.float32)  # BxCxK_
        C_mask = C_mask * mk_[:, None, :].repeat(1, C_mask.shape[1], 1)

        T_mask = T_inds_[:, None, :].repeat(1, T.shape[1], 1)
        temp_T_inds = torch.ones_like(T[:, :, 0]).cumsum(1)[:, :, None].repeat(1, 1, C_inds_.shape[1]) - 1
        T_mask = (T_mask == temp_T_inds).to(torch.float32)  # BxTxK_
        T_mask = T_mask * mk_[:, None, :].repeat(1, T_mask.shape[1], 1)

        U_ = self.relu(self.edge_init(U_)) * mk_[:, :, None].repeat(1, 1, self.nkernel)  #
        T_ = torch.sin(self.time_init(T))  # learned time embedding
        C_ = self.relu(self.chan_init(C_))  # embedding on one-hot encoded channel

        del temp_T_inds
        del temp_c_inds

        for i in range(self.n_layers):
            # channels as queries
            q_c = C_
            k_t = self.gather(T_, T_inds_)  # BxK_max x embd_len
            k = torch.cat([k_t, U_], -1)  # BxK_max x 2 * embd_len

            C__ = self.channel_time_attn[i](q_c, k, C_mask)  # attn (channel_embd, concat(time, values)) along with the mask

            # times as queries
            q_t = T_
            k_c = self.gather(C_, C_inds_)
            k = torch.cat([k_c, U_], -1)
            T__ = self.time_channel_attn[i](q_t, k, T_mask)

            # updating edge weights
            U_ = self.relu(U_ + self.edge_nn[i](torch.cat([U_, k_t, k_c], -1))) * mk_[:, :, None].repeat(1, 1, self.nkernel)

            # updating only channel nodes

            C_ = self.channel_attn[i](C__, C__)
            T_ = T__

        if x_repr_var is not None:
            score = torch.sigmoid(self.score_layer(x_repr_var))
            C_ = C_ + score * x_repr_var

        k_t = self.gather(T_, T_inds_)
        k_c = self.gather(C_, C_inds_)
        output = self.output(torch.cat([U_, k_t, k_c], -1))
        return output.squeeze(-1), target_mask_, source_, source_mask_, C__
