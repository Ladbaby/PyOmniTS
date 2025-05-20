import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

from utils.globals import logger

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionalEmbedding_ScaleFormer(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding_ScaleFormer, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, scale=1):
        return self.pe[:, scale:x.size(1)*scale+1:scale]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # print(f"{c_in=}")
        # print(f"{d_model=}")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        # TODO: update force overwrite
        c_in = 1024
        self.c_in = c_in
        logger.warning(f"FixedEmbedding currently use 1024 as max length, which may encounter error when input time step value is larger.")

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        if torch.any(x < 1).item():
            output = (self.emb((x * self.c_in).int()).squeeze(2)).detach()
            return output
        else:
            return self.emb(x.int()).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.general_embedding = FixedEmbedding(1, d_model)

        self.warning_counter = 0

    def forward(self, x_mark):
        # if (torch.any(x_mark > 1).item()) and (x_mark.shape[-1] in [4, 5]):
        if x_mark.shape[-1] in [4, 5]:
            '''
            if number of features is 4 or 5, then we assume it is a regular time series dataset
            '''
            try:
                x_mark = x_mark.long()

                minute_x = self.minute_embed(x_mark[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
                hour_x = self.hour_embed(x_mark[:, :, 3])
                weekday_x = self.weekday_embed(x_mark[:, :, 2])
                day_x = self.day_embed(x_mark[:, :, 1])
                month_x = self.month_embed(x_mark[:, :, 0])

                return hour_x + weekday_x + day_x + month_x + minute_x
            except Exception as e:
                if self.warning_counter == 0:
                    logger.warning(f"Maybe you are running datasets other than regular datasets. Temporal embedding with fallback to general embedding, where only x_mark's first value in 3rd feature dimension will be used")
                    self.warning_counter += 1

        # fall back to general temporal embedding if failed
        return self.general_embedding(x_mark[:, :, :1])


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        self.general_embed = nn.Linear(1, d_model, bias=False)

        self.warning_counter = 0

    def forward(self, x):
        try:
            return self.embed(x)
        except Exception as e:
            # fall back to general embedding if the dataset is not regular time series dataset
            if self.warning_counter == 0:
                logger.warning(f"Maybe you are running datasets other than regular datasets. TimeFeatureEmbedding with fallback to general embedding, where only x_mark's first value in 3rd feature dimension will be used")
                self.warning_counter += 1
            return self.general_embed(x[:, :, :1])

class TimeFeatureEmbedding_ScaleFormer(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding_ScaleFormer, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp+1, d_model, bias=False)
        self.general_embed = nn.Linear(1+1, d_model, bias=False)
        self.concats = dict()

        self.warning_counter = 0

    def forward(self, x, scale=1):
        if (scale, x.shape[0], x.shape[1]) not in self.concats:
            concat_tensor = torch.tensor([[[1/scale-0.5]]]).cuda().repeat(x.shape[0],x.shape[1],1)
            self.concats[(scale, x.shape[0], x.shape[1])] = concat_tensor
        else:
            concat_tensor = self.concats[(scale, x.shape[0], x.shape[1])]
        try:
            x = torch.cat((x, concat_tensor), 2)
            return self.embed(x)
        except Exception as e:
            # fall back to general embedding if the dataset is not regular time series dataset
            if self.warning_counter == 0:
                logger.warning(f"Maybe you are running datasets other than regular datasets. TimeFeatureEmbedding with fallback to general embedding, where only x_mark's first value in 3rd feature dimension will be used")
                self.warning_counter += 1
            x = torch.cat((x[:, :, :1], concat_tensor), 2)
            return self.general_embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, 
            embed_type=embed_type,
            freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, 
            embed_type=embed_type, 
            freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_ScaleFormer(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, is_decoder=False):
        super(DataEmbedding_ScaleFormer, self).__init__()
        if is_decoder:
            c_in += 1
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding_ScaleFormer(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding_ScaleFormer(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.is_decoder = is_decoder

    def forward(self, x, x_mark, scale, first_scale, label_len):
        if self.is_decoder:
            x = torch.cat((x, torch.ones((x.shape[0], x.shape[1], 1), device=x.device)), dim=2)
            if scale==first_scale:
                x[:,:label_len//scale,-1] = 0
                x[:,label_len//scale:,-1] = 0.5
            else:
                x[:,:label_len//scale,-1] = 0
                x[:,label_len//scale:,-1] = 1
        vembed = self.value_embedding(x)
        pembed = self.position_embedding(x, scale)
        tembed = self.temporal_embedding(x_mark, scale)
        # fix for irregular time series datasets
        min_length = min(vembed.shape[1], pembed.shape[1], tembed.shape[1])
        x = vembed[:, :min_length] + pembed[:, :min_length] + tembed[:, :min_length]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, 
            embed_type=embed_type,
            freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, 
            embed_type=embed_type, 
            freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x.shape=[128, 96, 2]; x_mark.shape=[128, 96, 4]
        # logger.debug(f"{x.shape=}")
        # logger.debug(f"{x_mark.shape=}")
        # logger.debug(f"{self.value_embedding(x).shape=}")
        # logger.debug(f"{self.temporal_embedding(x_mark).shape=}")
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x = self.value_embedding(x)
        # x: [Batch Variate d_model]
        return self.dropout(x)