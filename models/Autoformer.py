import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

from layers.Formers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp
from layers.Formers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Formers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021)
    - paper link: https://openreview.net/pdf?id=I55UqU-M11y
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.output_attention = configs.output_attention
        self.configs = configs

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * self.seq_len, configs.n_classes)
        else:
            raise NotImplementedError

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        # print(f"{trend_init.shape=}")
        # print(f"{seasonal_init.shape=}")
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # print(f"{trend_init.shape=}")
        # print(f"{seasonal_init.shape=}")
        # enc
        # print(f"{x_enc.shape=}")
        # print(f"{x_mark_enc.shape=}")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        # print(f"{seasonal_init.shape=}")
        # print(f"{x_mark_dec.shape=}")
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

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

        x_dec = torch.cat([x[:, -self.label_len:, :], torch.zeros_like(y).float()], dim=1).float().to(x.device)
        x_mark_dec = torch.cat([x_mark[:, -self.label_len:, :], y_mark], dim=1).float().to(x_mark.device)
        # END adaptor


        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x, x_mark, x_dec, x_mark_dec)
            # handle the special Multi-variate to Single-variate forecast task
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        elif self.task_name == 'classification':
            dec_out = self.classification(x, x_mark)
            return {
                "pred_class": dec_out,
                "true_class": y_class
            }
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x, x_mark)
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        else:
            raise NotImplementedError
        # elif self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x)
        #     return dec_out  # [B, L, D]
        # else:
        #     return None

        
