# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.Raindrop import BackboneRaindrop
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs


class Model(nn.Module):
    """
    - Paper: "Graph-Guided Network for Irregularly Sampled Multivariate Time Series" (ICLR 2022)
    - Paper link: https://openreview.net/forum?id=Kwm8I7dU-l5
    - Code adapted from: https://github.com/WenjieDu/PyPOTS

    The core wrapper assembles the submodules of Raindrop classification model
    and takes over the forward progress of the algorithm.
    """
    def __init__(
        self,
        configs: ExpConfigs,
    ):
        super().__init__()
        self.configs = configs
        self.d_ob = 4 if configs.task_name == "classification" else configs.d_model
        n_features = self.configs.enc_in
        n_layers = configs.n_layers # 2
        d_model = n_features * self.d_ob
        n_heads = self.configs.n_heads
        d_ffn = 2 * d_model
        n_classes = self.configs.n_classes # warning: not implemented
        dropout = configs.dropout # 0.3
        max_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        d_static = 9
        aggregation = "mean"
        sensor_wise_mask = False 
        static = False

        d_pe = 16
        self.d_pe = d_pe
        self.aggregation = aggregation
        self.sensor_wise_mask = sensor_wise_mask
        self.n_features = n_features

        self.backbone = BackboneRaindrop(
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_ffn,
            n_classes,
            dropout,
            max_len,
            d_static,
            d_pe,
            aggregation,
            sensor_wise_mask,
            static,
        )

        if static:
            d_final = d_model + n_features
        else:
            d_final = d_model + d_pe


        if configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            self.mlp_forecast = nn.Sequential(
                nn.Linear(self.d_ob, self.d_ob),
                nn.ReLU(),
                nn.Linear(self.d_ob, configs.pred_len_max_irr if configs.pred_len_max_irr is not None else configs.pred_len),
            )
        elif configs.task_name == "classification":
            self.mlp_static = nn.Sequential(
                nn.Linear(d_final, d_final),
                nn.ReLU(),
                nn.Linear(d_final, n_classes),
            )
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        static=None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, dtype=x.dtype, device=x.device)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, dtype=y.dtype, device=y.device)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        timestamps = x_mark[:, :, 0] # (BATCH_SIZE, SEQ_LEN)
        lengths = self.compute_lengths(timestamps) # (BATCH_SIZE)
        missing_mask = x_mask # (BATCH_SIZE, SEQ_LEN, ENC_IN)
        X = x
        device = X.device
        # END adaptor

        # representation: (SEQ_LEN, BATCH_SIZE, ENC_IN * d_ob + d_pe)
        # mask: (BATCH_SIZE, SEQ_LEN)
        representation, mask = self.backbone(
            X,
            timestamps,
            lengths,
        )

        lengths2 = lengths.unsqueeze(1).to(device) # (BATCH_SIZE, 1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long() # (SEQ_LEN, BATCH_SIZE, 1)
        if self.configs.task_name in ["long_term_forecast", "short_term_forecast", "imputation"]:
            output = representation.mean(0).squeeze(0)
        elif self.configs.task_name == "classification":
            if self.sensor_wise_mask:
                output = torch.zeros([BATCH_SIZE, self.n_features, self.d_ob + self.d_pe], device=device)
                extended_missing_mask = missing_mask.view(-1, BATCH_SIZE, self.n_features)
                for se in range(self.n_features):
                    representation = representation.view(-1, BATCH_SIZE, self.n_features, (self.d_ob + self.d_pe))
                    out = representation[:, :, se, :]
                    l_ = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)  # length
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (l_ + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.n_features * (self.d_ob + self.d_pe)]) # (BATCH_SIZE, ENC_IN * (d_ob + d_pe))
            elif self.aggregation == "mean":
                output = torch.sum(representation * (1 - mask2), dim=0) / (lengths2 + 1) # (BATCH_SIZE, ENC_IN * d_ob + d_pe)
            else:
                raise RuntimeError

        if self.configs.task_name == "classification":
            if static is not None:
                emb = self.static_emb(static)
                output = torch.cat([output, emb], dim=1)
            logits = self.mlp_static(output) # (BATCH_SIZE, n_classes)
            return {
                "pred_class": logits,
                "true_class": y_class,
            }
        elif self.configs.task_name in ["long_term_forecast", "short_term_forecast", "imputation"]:
            output = rearrange(output[:, :ENC_IN * self.d_ob], "BATCH_SIZE (ENC_IN D) -> BATCH_SIZE ENC_IN D", BATCH_SIZE=BATCH_SIZE, ENC_IN=self.n_features, D=self.d_ob)
            output = rearrange(self.mlp_forecast(output), "BATCH_SIZE ENC_IN PRED_LEN -> BATCH_SIZE PRED_LEN ENC_IN")
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": output[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError

    def compute_lengths(self, timestamps: Tensor):
        '''
        `lengths` of shape (BATCH_SIZE) is the actual time lengths of samples
        '''
        # Create a mask indicating non-zero elements
        mask = timestamps != 0
        # Generate indices for each position in the sequence
        indices = torch.arange(timestamps.size(1), device=timestamps.device).expand_as(timestamps)
        # Compute the maximum index for each row where the element is non-zero
        max_indices = (indices * mask).max(dim=1, keepdim=True)[0]
        # Check if there are any non-zero elements in each row
        any_non_zero = mask.any(dim=1, keepdim=True)
        # Calculate lengths, adjusting for all-zero rows
        lengths = (max_indices + 1) * any_non_zero
        return lengths.squeeze(-1)