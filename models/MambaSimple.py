import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, einsum

from layers.Formers.Embed import DataEmbedding
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs


class Model(nn.Module):
    """
    Another implementation of Mamba
    
    - code adapted from: https://github.com/johnma2006/mamba-minimal/
    """

    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        if self.pred_len > self.seq_len:
            logger.error(f"MambaSimple will encounter error when {self.pred_len=} > {self.seq_len=}")

        self.d_inner = configs.d_model * configs.mamba_expand
        self.dt_rank = math.ceil(configs.d_model / 16)

        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.layers = nn.ModuleList([ResidualBlock(configs, self.d_inner, self.dt_rank) for _ in range(configs.e_layers)])
        self.norm = RMSNorm(configs.d_model)

        if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)
        elif configs.task_name == "classification":
            self.decoder_classification = nn.Linear(configs.d_model, configs.n_classes)
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        y: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        x_enc = x
        x_mark_enc = x_mark
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            mean_enc = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - mean_enc
            std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_enc = x_enc / std_enc

        x_enc = self.embedding(x_enc, x_mark_enc)
        for layer in self.layers:
            x_enc = layer(x_enc)

        x_enc = self.norm(x_enc)

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            x_out = self.out_layer(x_enc)
            x_out = x_out * std_enc + mean_enc
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": x_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.decoder_classification(x_enc.mean(1)),
                "true_class": y_class
            }
        else:
            raise NotImplementedError


class ResidualBlock(nn.Module):
    def __init__(self, configs, d_inner, dt_rank):
        super(ResidualBlock, self).__init__()
        
        self.mixer = MambaBlock(configs, d_inner, dt_rank)
        self.norm = RMSNorm(configs.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    def __init__(self, configs: ExpConfigs, d_inner, dt_rank):
        super(MambaBlock, self).__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(configs.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            bias = True,
            kernel_size = configs.mamba_d_conv,
            padding = configs.mamba_d_conv - 1,
            groups = self.d_inner
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + configs.d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, configs.d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, configs.d_model, bias=False)

    def forward(self, x):
        """
        Figure 3 in Section 3.4 in the paper
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x) # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """
        Algorithm 2 in Section 3.2 in the paper
        """
        
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float()) # [d_in, n]
        D = self.D.float() # [d_in]

        x_dbl = self.x_proj(x) # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1) # delta: [B, L, d_rank]; B, C: [B, L, n]
        delta = F.softplus(self.dt_proj(delta)) # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n")) # A is discretized using zero-order hold (ZOH) discretization
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n") # B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors: "A is the more important term and the performance doesn't change much with the simplification on B"

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1) # [B, L, d_in]
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
