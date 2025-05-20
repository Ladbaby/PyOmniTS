import torch
import torch.nn as nn
from torch import Tensor

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    - paper: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/26317
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        else:
            self.pred_len = configs.pred_len_max_irr or configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # self.Linear_Seasonal = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        # self.Linear_Trend = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

        if configs.task_name == "classification":
            self.decoder_classification = nn.Linear(self.seq_len * self.enc_in, configs.n_classes)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model) d_model is enc_in ?
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.decoder_classification(output)
        return output

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

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x)
            # handle the special Multi-variate to Single-variate forecast task
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        elif self.task_name == "classification":
            out = self.classification(x)
            return {
                "pred_class": out,
                "true_class": y_class
            }
        else:
            raise NotImplementedError
        # elif self.task_name == 'imputation':
        #     dec_out = self.imputation(x)
        #     return dec_out  # [B, L, D]
        # elif self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x)
        #     return dec_out  # [B, L, D]
        # elif self.task_name == 'classification':
        #     dec_out = self.classification(x)
        #     return dec_out  # [B, N]
        # else:
        #     return None

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x_enc, **kwargs):
        # padding on the both ends of time series
        front = x_enc[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x_enc[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_enc = torch.cat([front, x_enc, end], dim=1)
        x_enc = self.avg(x_enc.permute(0, 2, 1))
        x_enc = x_enc.permute(0, 2, 1)
        return x_enc


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

