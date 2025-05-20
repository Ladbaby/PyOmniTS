import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

from layers.Formers.Embed import DataEmbedding
from layers.Formers.Autoformer_EncDec import series_decomp, series_decomp_multi
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    - paper: "MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting" (ICLR 2023)
    - paper link: https://openreview.net/pdf?id=zt53IDUR1U
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """
    def __init__(self, configs: ExpConfigs, conv_kernel=[12, 16]):
        """
        conv_kernel: downsampling and upsampling convolution kernel_size
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.label_len = configs.label_len

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((self.seq_len + self.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((self.seq_len + self.pred_len + ii - 1) // ii)


        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model, n_heads=configs.n_heads,
                                             dropout=configs.dropout,
                                             d_layers=configs.d_layers, decomp_kernel=decomp_kernel,
                                             c_out=configs.c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # refer to DLinear
            self.regression = nn.Linear(self.seq_len, self.pred_len)
            self.regression.weight = nn.Parameter(
                (1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
                requires_grad=True)
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.c_out * self.seq_len, configs.n_classes)

    def forecast(self, x_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.label_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)

        # embedding
        dec_out = self.dec_embedding(seasonal_init_enc, x_mark_dec)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend
        return dec_out

    def anomaly_detection(self, x_enc):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)

        # embedding
        dec_out = self.dec_embedding(seasonal_init_enc, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        # embedding
        dec_out = self.dec_embedding(seasonal_init_enc, x_mark_enc)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out + trend

        # Output from Non-stationary Transformer
        output = self.act(dec_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

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

        x_dec = torch.cat([x[:, -self.configs.label_len:, :], torch.zeros_like(y).float()], dim=1).float().to(x.device)
        x_mark_dec = torch.cat([x_mark[:, -self.configs.label_len:, :], y_mark], dim=1).float().to(x_mark.device)
        # END adaptor

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x, x_dec, x_mark_dec)
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

class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=input.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

            # merge
        mg = torch.tensor([], device=src.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)
        
