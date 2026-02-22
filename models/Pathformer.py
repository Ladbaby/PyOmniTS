# Code from: https://github.com/Ladbaby/PyOmniTS
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from layers.Pathformer.layers.AMS import AMS
from layers.RevIN.RevIN import RevIN
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting" (ICLR 2024)
    - paper link: https://openreview.net/forum?id=lJkOCMP2aW
    - code adapted from: https://github.com/decisionintelligence/pathformer
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        self.layer_nums = configs.n_layers  # 设置pathway的层数
        self.num_nodes = configs.enc_in
        self.pre_len = self.pred_len
        self.seq_len = configs.seq_len_max_irr or configs.seq_len

        self.layer_nums = 1
        self.k = 1 # top-k patch size at every layer
        self.num_experts_list = [1]
        self.patch_size_list = np.array([configs.patch_len_max_irr or configs.patch_len]).reshape(self.layer_nums, -1).tolist()

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = 1
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.batch_norm = 0

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(
        self, 
        x: Tensor,
        y: Tensor | None = None,
        y_mask: Tensor | None = None,
        y_class: Tensor | None = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        # END adaptor

        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))


        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
                "loss2": balance_loss
            }
        else:
            raise NotImplementedError()


