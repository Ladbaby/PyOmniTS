import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

#TiDE
class Model(nn.Module):  
    """
    - paper: "Long-term Forecasting with TiDE: Time-series Dense Encoder" (TMLR 2023)
    - paper link: https://openreview.net/forum?id=pCbC3aQB5W
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """
    def __init__(self, configs: ExpConfigs, bias=True, feature_encode_dim=2): 
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len  #H 
        self.hidden_dim=configs.d_model
        self.res_hidden=configs.d_model 
        self.encoder_num=configs.e_layers
        self.decoder_num=configs.d_layers
        self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden=configs.d_ff
        dropout=configs.dropout

        
        # freq_map = {'h': 4, 't': 5, 's': 6,
        #             'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        
        # self.feature_dim=freq_map[self.freq]
        self.feature_dim = configs.enc_in


        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        elif self.task_name == 'imputation':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        elif self.task_name == 'anomaly_detection':
            self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias))
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        elif self.task_name == "classification":
            self.decoder_classification = nn.Linear(self.hidden_dim * configs.enc_in, configs.n_classes)
        else:
            raise NotImplementedError
        
    def forecast(self, x_enc, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,self.seq_len:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
        
        
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        feature = self.feature_encoder(x_mark_enc)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.seq_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:,:self.seq_len], decoded], dim=-1)).squeeze(-1) + self.residual_proj(x_enc)
    
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.seq_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.seq_len))
        return dec_out
    
    def classification(self, x_enc, x_mark_enc):
        BATCH_SIZE = x_enc.size(0)
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        feature = self.feature_encoder(x_mark_enc)
        hidden_list = []
        # TiDE only accepts single variate input
        for variable in range(x_enc.shape[-1]):
            hidden = self.encoders(torch.cat([x_enc[:, :, variable], feature.reshape(feature.shape[0], -1)], dim=-1))
            hidden_list.append(hidden)

        output = torch.stack(hidden_list, dim=-1).reshape(BATCH_SIZE, -1)

        return self.decoder_classification(output)
    
    
    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
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
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        x_mark[x_mark == 1] = 0.9999 # cannot process value == 1 in mark
        y_mark[y_mark == 1] = 0.9999

        y_mark = repeat(y_mark[:, :, :1], "B L 1 -> B L F", F=x.shape[-1])
        x_mark = repeat(x_mark[:, :, :1], "B L 1 -> B L F", F=x.shape[-1])
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        batch_y_mark=torch.concat([x_mark, y_mark[:, -self.pred_len:, :]],dim=1)
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = torch.stack([self.forecast(x[:, :, feature], batch_y_mark) for feature in range(x.shape[-1])],dim=-1)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.classification(x, x_mark),
                "true_class": y_class
            }
        else:
            raise NotImplementedError

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


