import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.RevIN.RevIN import RevIN
from layers.Leddam.Leddam import Leddam
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(nn.Module):
    '''
    - paper: "Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling" (ICML 2024)
    - paper link: https://openreview.net/forum?id=87CYNyCGOo
    - code adapted from: https://github.com/Levi-Ackman/Leddam
    '''
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.revin = True if configs.task_name != "classification" else False
        self.revin_layer = RevIN(num_features=configs.enc_in)
        self.leddam = Leddam(configs.enc_in, configs.seq_len_max_irr or configs.seq_len, configs.d_model,
                             configs.dropout, "sincos", kernel_size=25, n_layers=configs.n_layers)

        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.Linear_main = nn.Linear(configs.d_model, self.pred_len)
            self.Linear_res = nn.Linear(configs.d_model, self.pred_len)
            self.Linear_main.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([self.pred_len, configs.d_model]))
            self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([self.pred_len, configs.d_model]))
        elif configs.task_name == "classification":
            self.decoder_classification = nn.Linear(configs.d_model * configs.enc_in, configs.n_classes)

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
            y_mask = torch.ones_like(y, dtype=y.dtype, device=y.device)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        # END adaptor

        if self.revin:
            x = self.revin_layer(x, "norm")
        res, main = self.leddam(x)

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            main_out = self.Linear_main(main.permute(0, 2, 1)).permute(0, 2, 1)
            res_out = self.Linear_res(res.permute(0, 2, 1)).permute(0, 2, 1)
            pred = main_out+res_out
            if self.revin:
                pred = self.revin_layer(pred, "denorm")
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": pred[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        elif self.configs.task_name == "classification":
            output = self.decoder_classification(rearrange(main + res, "B D ENC_IN -> B (ENC_IN D)"))
            return {
                "pred_class": output,
                "true_class": y_class
            }
        else:
            raise NotImplementedError
