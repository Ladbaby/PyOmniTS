import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

from layers.Formers.Embed import DataEmbedding
from layers.ETSformer.ETSformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, Transform
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "ETSformer: Exponential Smoothing Transformers for Time-series Forecasting"
    - paper link: https://arxiv.org/abs/2202.01381
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """

    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        assert configs.e_layers == configs.d_layers, "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.enc_in, self.seq_len, self.pred_len, configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs.d_model, configs.n_heads, configs.c_out, self.pred_len,
                    dropout=configs.dropout,
                ) for _ in range(configs.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

        if configs.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.decoder_classification = nn.Linear(self.seq_len * configs.d_model, configs.n_classes)

    def forecast(self, x_enc, x_mark_enc):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def anomaly_detection(self, x_enc):
        res = self.enc_embedding(x_enc, None)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def classification(self, x_enc, x_mark_enc):
        res = self.enc_embedding(x_enc, None)
        _, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growths = torch.sum(torch.stack(growths, 0), 0)[:, :self.seq_len, :]
        seasons = torch.sum(torch.stack(seasons, 0), 0)[:, :self.seq_len, :]

        enc_out = growths + seasons
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)

        # Output
        output = output * x_mark_enc  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.decoder_classification(output)  # (batch_size, num_classes)
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

        x_mark[x_mark == 1] = 0.9999
        # END adaptor

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x, x_mark)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.task_name == "classification":
            out = self.classification(x, x_mark)
            return {
                "pred_class": out,
                "true_class": y_class
            }
        else:
            raise NotImplementedError
