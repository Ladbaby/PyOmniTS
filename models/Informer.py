import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

from layers.Formers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.Formers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Formers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """
    - paper: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)
    - paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325
    - code adapted from: https://github.com/thuml/Time-Series-Library
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
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
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.informer_distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        elif self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.n_classes)
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor = None, 
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        y_class: Tensor = None,
        enc_self_mask=None, 
        dec_self_mask=None, 
        dec_enc_mask=None, 
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

        if self.configs.task_name == "short_term_forecast":
            # Normalization
            mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
            x = x - mean_enc
            std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
            x = x / std_enc

        if self.configs.task_name == "classification":
            enc_out = self.enc_embedding(x, None)
        else:
            enc_out = self.enc_embedding(x, x_mark)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if self.configs.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            # handle the special Multi-variate to Single-variate forecast task
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        elif self.configs.task_name == "classification":
            output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
            output = self.dropout(output)
            output = output * x_mark  # zero-out padding embeddings
            output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
            output = self.projection(output)  # (batch_size, num_classes)
            return {
                "pred_class": output,
                "true_class": y_class
            }
        elif self.configs.task_name == "imputation":
            dec_out = self.projection(enc_out)
            # handle the special Multi-variate to Single-variate forecast task
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": dec_out[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        else:
            raise NotImplementedError