import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    '''
    - paper: "TSMixer: An All-MLP Architecture for Time Series Forecasting" (TMLR)
    - paper link: https://openreview.net/forum?id=wbpxTuXgm0
    - code adapted from: https://github.com/thuml/Time-Series-Library
    '''
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(self.seq_len, self.pred_len)
        elif configs.task_name == "classification":
            self.projection = nn.Linear(self.seq_len * configs.enc_in, configs.n_classes)
        else:
            raise NotImplementedError

    def forecast(self, x_enc):
        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return enc_out

    def classification(self, x_enc):
        BATCH_SIZE = x_enc.size(0)
        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.reshape(BATCH_SIZE, -1))

        return enc_out

    def forward(
        self, 
        x: Tensor,
        y: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
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
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.classification(x),
                "true_class": y_class
            }
        else:
            raise NotImplementedError

class ResBlock(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(ResBlock, self).__init__()

        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len

        self.temporal = nn.Sequential(
            nn.Linear(self.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
