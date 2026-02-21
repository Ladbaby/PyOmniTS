import math

import torch
import torch.nn as nn
from einops import *
from torch import Tensor

from layers.ReIMTS.layers.Formers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from layers.ReIMTS.layers.Formers.Transformer_EncDec import Encoder as FormerEncoder
from layers.ReIMTS.layers.Formers.Transformer_EncDec import (
    EncoderLayer as FormerEncoderLayer,
)
from layers.ReIMTS.layers.TimeCHEAT.graph_layer import Encoder
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis" (AAAI 2025)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/34076
    - code adapted from: https://github.com/Alrash/TimeCHEAT
    """
    def __init__(
        self,
        configs: ExpConfigs, 
        current_level: int, 
        time_len_list: list[int], 
        **kwargs
    ):
        super().__init__()
        self.current_level = current_level
        self.total_scale_factor = 1
        for i in range(len(time_len_list)):
            if i == len(time_len_list) - 1:
                break
            else:
                self.total_scale_factor *= math.ceil(time_len_list[i] / time_len_list[i + 1])
        self.n_fractal_levels = len(time_len_list)

        # BEGIN adaptor
        self.configs = configs
        channels = configs.enc_in
        attn_head = configs.n_heads
        latent_dim = configs.d_model
        n_layers = configs.n_layers

        # assert configs.seq_len % configs.patch_len == 0, f"seq_len {configs.seq_len} should be divisible by patch_len {configs.patch_len}"
        # n_patches: int = configs.seq_len // configs.patch_len
        n_patches: int = 1 # force overwrite as 1 when used as backbone, since no need to split inside backbone

        ref_points_per_patch = 32
        ref_points = ref_points_per_patch * n_patches
        dropout = configs.dropout
        former_factor = configs.factor
        former_dff = configs.d_ff
        former_output_attention = False
        former_layers = 3
        former_heads = 8
        former_activation = "gelu"
        downstream = configs.task_name

        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len
        # END adaptor

        self.dim = channels
        self.ath = attn_head
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.n_patches = n_patches
        self.register_buffer('patch_range', torch.linspace(0, self.seq_len, self.n_patches + 1))
        if ref_points % self.n_patches != 0:
            logger.error(f"--mtan_num_ref_points {ref_points} should be divisible by n_patches {self.n_patches}", stack_info=True)
            exit(1)
        self.register_buffer('ref_points', torch.linspace(0, self.seq_len, ref_points))
        self.ref_points = self.ref_points.reshape(n_patches, -1)

        # graph patch
        self.encoder = Encoder(dim=self.dim, attn_head=self.ath, n_patches=self.n_patches, nkernel=self.latent_dim, n_layers=self.n_layers)
        self.position_embedding = PositionalEmbedding(self.ref_points.size(-1))
        self.dropout = nn.Dropout(dropout)

        # transformer
        self.former = FormerEncoder(
            [
                FormerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, former_factor, attention_dropout=dropout,
                                      output_attention=former_output_attention), self.ref_points.size(-1), former_heads),
                    self.ref_points.size(-1),
                    former_dff,
                    dropout=dropout,
                    activation=former_activation
                ) for _ in range(former_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.ref_points.size(-1))
        )

        self.ds = downstream.lower()
        if self.ds in ['classification']:
            self.flatten = nn.Flatten(start_dim=-2)
            self.pj_dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(self.n_patches * self.ref_points.size(-1) * self.dim, configs.n_classes)
        elif self.ds in ['short_term_forecast', 'long_term_forecast', 'imputation']:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.n_patches * self.ref_points.size(-1) * self.total_scale_factor, self.pred_len * self.total_scale_factor)
            self.dropout = nn.Dropout(dropout)
        else:
            raise NotImplementedError()

    def _split_patch(self, data, mask, time, i_patch):
        start, end, ref_points = self.patch_range[i_patch], self.patch_range[i_patch + 1], self.ref_points[i_patch].to(data.device)
        time_mask = torch.logical_and(torch.logical_and(time >= start, time <= end), mask.sum(-1) > 0)
        num_observed = time_mask.sum(1).long()

        n_ref_points = ref_points.size(0)
        patch = torch.zeros(data.size(0), num_observed.max() + n_ref_points, self.dim, device=data.device)
        patch_mask, patch_time = torch.zeros_like(patch), torch.zeros(patch.size(0), patch.size(1), device=data.device)
        rp_mask, indices = torch.zeros_like(patch_mask), torch.arange(patch.size(1)).to(data.device)
        for i in range(data.size(0)):
            patch_mask[i, :num_observed[i], :] = mask[i, time_mask[i]]
            patch[i, :num_observed[i], :] = data[i, time_mask[i]]
            patch_time[i, :num_observed[i]] = time[i, time_mask[i]]

            # insert ref points
            patch_time[i, num_observed[i]: num_observed[i] + n_ref_points] = ref_points
            sorted_index = torch.cat([torch.argsort(patch_time[i, :num_observed[i] + n_ref_points]), indices[num_observed[i] + n_ref_points:]])
            patch_mask[i] = patch_mask[i, sorted_index]
            patch[i] = patch[i, sorted_index]
            patch_time[i] = patch_time[i, sorted_index]
            rp_mask[i, sorted_index[num_observed[i]:num_observed[i] + n_ref_points]] = 1.
        # return patch.clone(), patch_mask.clone(), patch_time.clone(), rp_mask.clone()
        return patch, patch_mask, patch_time, rp_mask

    # def embedding(self, data):
    def embedding(
        self,
        vals: Tensor,
        mask: Tensor,
        time: Tensor,
        x_repr_var: Tensor
    ):
        # vals, mask, time = data[..., :self.dim], data[..., self.dim:-1], data[..., -1]

        # encoder
        repr_patch = []
        for i_patch in range(self.n_patches):
            v, m, t, rp_m = self._split_patch(vals, mask, time, i_patch)

            v = v * m
            context_mask = m + rp_m
            # out -> n_patch * B * {t_i} * channel * latent_dim
            repr, repr_mask, _, _, pred_repr_var = self.encoder(t, v, context_mask, rp_m, i_patch, x_repr_var)
            # repr_patch.append(repr[repr_mask.sum(-1) > 0, ...].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
            repr_patch.append(repr[repr_mask == 1].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
        # combined, [batch x patch_num x patch_len x dim] => [batch x dim x patch_num x patch_len]
        repr_patch = torch.cat(repr_patch, dim=1).contiguous().permute(0, 3, 1, 2)

        # positional embedding
        repr_patch = torch.reshape(repr_patch, (repr_patch.shape[0] * repr_patch.shape[1], repr_patch.shape[2], repr_patch.shape[3]))

        # transformer encode
        embedding = repr_patch
        # [batch x dim x patch_num x patch_len] => [batch x dim x patch_len x patch_num]
        embedding = torch.reshape(embedding, (-1, self.dim, embedding.shape[-2], embedding.shape[-1])).permute(0, 1, 3, 2)
        return embedding, pred_repr_var

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        x_repr_var: Tensor = None,
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN),dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        x_mark = x_mark[:, :, 0]
        y_mark = y_mark[:, :, 0]
        # END adaptor

        embedding, pred_repr_var = self.embedding(
            vals=x,
            mask=x_mask,
            time=x_mark,
            x_repr_var=x_repr_var
        )

        if self.current_level != self.n_fractal_levels - 1:
            return {
                "pred_repr_var": pred_repr_var
            }
        else:
            if self.ds in ['classification']:
                # temporally comment out the unfinished adaptation for classification
                out = self.dropout(self.flatten(embedding)).reshape(embedding.shape[0], -1)
                out = self.projection(out)
                return {
                    "pred_class": out,
                    "true_class": y_class,
                    "pred_repr_var": pred_repr_var
                }
            elif self.ds in ['short_term_forecast', 'long_term_forecast', 'imputation']:
                # BEGIN adaptor
                y = y[::self.total_scale_factor]
                y_mark = y_mark[::self.total_scale_factor]
                y_mask = y_mask[::self.total_scale_factor]
                # END adaptor
                out = self.dropout(self.linear(self.unpatchify_variable_embedding(self.flatten(embedding))))
                out = out.permute(0, 2, 1)

                f_dim = -1 if self.configs.features == 'MS' else 0
                PRED_LEN = y.shape[1]
                return {
                    "pred": out[:, -PRED_LEN:, f_dim:],
                    "true": y[:, :, f_dim:],
                    "mask": y_mask[:, :, f_dim:],
                    "pred_repr_var": pred_repr_var
                }
            else:
                raise NotImplementedError()

    def unpatchify_variable_embedding(self, x):
        return rearrange(x, "(B N_PATCH) ENC_IN D -> B ENC_IN (N_PATCH D)", N_PATCH=self.total_scale_factor)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]