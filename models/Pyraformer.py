import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.Pyraformer.Pyraformer_EncDec import Encoder
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """ 
    - paper: "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting" (ICLR 2022)
    - paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """

    def __init__(self, configs: ExpConfigs, inner_size=5):
        """
        window_size: list, the downsample window size in pyramidal attention.
        inner_size: int, the size of neighbour attention
        """
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.patch_len = configs.patch_len_max_irr or configs.patch_len
        self.d_model = configs.d_model
        self.task_name = configs.task_name

        window_size = [self.patch_len]

        self.encoder = Encoder(configs, window_size, inner_size)

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, self.pred_len * configs.enc_in)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model, configs.enc_in, bias=True)
        elif self.task_name == 'classification':
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                (len(window_size)+1)*self.d_model * self.seq_len, configs.n_classes)
        else:
            raise NotImplementedError

    def long_forecast(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out
    
    def short_forecast(self, x_enc, x_mark_enc):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.encoder(x_enc, None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output

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

        x_mark[x_mark == 1] = 0.9999 # cannot process value == 1 in mark
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.configs.task_name == 'long_term_forecast':
                dec_out = self.long_forecast(x, x_mark)
            elif self.configs.task_name == 'short_term_forecast':
                dec_out = self.short_forecast(x, x_mark)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            output = self.classification(x, x_mark)
            return {
                "pred_class": output,
                "true_class": y_class
            }
        else:
            raise NotImplementedError
