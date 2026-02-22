# Code from: https://github.com/Ladbaby/PyOmniTS
# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autoformer (https://arxiv.org/pdf/2106.13008.pdf) implementation
# from https://github.com/thuml/Autoformer by THUML @ Tsinghua University
####################################################################################

import torch
import torch.nn as nn
from einops import *
from torch import Tensor

from layers.Formers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Formers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from layers.Formers.Embed import DataEmbedding_ScaleFormer
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    Multi-scale version of Autoformer

    - paper: "Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting" (ICLR 2023)
    - paper link: https://openreview.net/forum?id=sCrnllCtjoE
    - code adapted from: https://github.com/BorealisAI/scaleformer
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)

        # Embedding
        # We use our new DataEmbedding which incldues the scale information
        self.enc_embedding = DataEmbedding_ScaleFormer(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_ScaleFormer(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout, is_decoder=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg = configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        """
        following functions will be used to manage scales
        """
        self.scale_factor = configs.scale_factor
        self.scales = configs.scaleformer_scales
        self.mv = moving_avg()
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='linear')
        self.input_decomposition_type = 1

        if configs.task_name == "classification":
            logger.exception("AutoformerMS (Scaleformer) does not support classification task!", stack_info=True)
            exit(1)

    def forward(
        self, 
        x: Tensor,  
        x_mark: Tensor = None, 
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        x_mark[x_mark == 1] = 0.9999 # cannot process value == 1 in mark
        y_mark[y_mark == 1] = 0.9999

        x_dec = torch.cat([x[:, -self.label_len:, :], torch.zeros_like(y).float()], dim=1).float().to(x.device)
        x_mark_dec = torch.cat([x_mark[:, -self.label_len:, :], y_mark], dim=1).float().to(x_mark.device)
        # END adaptor

        scales = self.scales
        label_len = x_dec.shape[1]-self.pred_len
        outputs = []
        for scale in scales:
            enc_out = self.mv(x, scale)
            if scale == scales[0]: # initialize the input of decoder at first step
                if self.input_decomposition_type == 1:
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    tmp_mean = torch.mean(enc_out, dim=1).unsqueeze(1).repeat(1, self.pred_len//scale, 1)
                    zeros = torch.zeros([x_dec.shape[0], self.pred_len//scale, x_dec.shape[2]], device=x.device)
                    seasonal_init, trend_init = self.decomp(enc_out)
                    trend_init = torch.cat([trend_init[:, -self.label_len//scale:, :], tmp_mean], dim=1)
                    seasonal_init = torch.cat([seasonal_init[:, -self.label_len//scale:, :], zeros], dim=1)
                    dec_out = self.mv(x_dec, scale) - mean
                else:
                    dec_out = self.mv(x_dec, scale)
                    mean = enc_out.mean(1).unsqueeze(1)
                    enc_out = enc_out - mean
                    dec_out[:, :label_len//scale, :] = dec_out[:, :label_len//scale, :] - mean
            else: # generation the input at each scale and cross normalization
                dec_out = self.upsample(dec_out_coarse.detach().permute(0,2,1)).permute(0,2,1)
                dec_out[:, :label_len//scale, :] = self.mv(x_dec[:, :label_len, :], scale)
                mean = torch.cat((enc_out, dec_out[:, label_len//scale:, :]), 1).mean(1).unsqueeze(1)
                enc_out = enc_out - mean
                dec_out = dec_out - mean

            # redefining the inputs to the decoder to be scale aware
            trend_init = torch.zeros_like(dec_out)
            seasonal_init = dec_out

            enc_out = self.enc_embedding(enc_out, x_mark[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            enc_out, attns = self.encoder(enc_out)
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec[:, scale//2::scale], scale=scale, first_scale=scales[0], label_len=label_len)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
            dec_out_coarse = seasonal_part + trend_part

            dec_out_coarse = dec_out_coarse + mean
            outputs.append(dec_out_coarse[:, -self.pred_len//scale:, :])
        
        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast', "imputation"]:
            # handle the special Multi-variate to Single-variate forecast task
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": outputs[-1][:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        else:
            raise NotImplementedError()

class moving_avg(nn.Module):
    """
    Downsample series using an average pooling
    """
    def __init__(self):
        super(moving_avg, self).__init__()

    def forward(self, x, scale=1):
        if x is None:
            return None
        x = nn.functional.avg_pool1d(x.permute(0, 2, 1), scale, scale)
        x = x.permute(0, 2, 1)
        return x
