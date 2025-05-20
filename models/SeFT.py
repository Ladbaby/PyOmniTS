from typing import Sequence, Dict, List, Tuple
from collections.abc import Sequence as SequenceType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from einops import *

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    - paper: "Set Functions for Time Series" (ICML 2020)
    - paper link: https://proceedings.mlr.press/v119/horn20a.html
    - code adapted from: https://github.com/mims-harvard/Raindrop
    
        original code https://github.com/BorgwardtLab/Set_Functions_for_Time_Series written in Tensorflow
    """

    def __init__(
        self, 
        configs: ExpConfigs,
    ):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        # BEGIN adaptor
        d_inp = configs.enc_in
        nhid = 128
        nlayers = configs.n_layers
        dropout = configs.dropout
        max_len = configs.seq_len_max_irr or configs.seq_len
        d_static = None
        MAX = 10000
        aggreg = "mean"
        static = False
        self.static = static
        # END adaptor


        self.model_type = 'Transformer'

        d_pe = 16
        d_enc = d_inp

        self.pos_encoder = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_value = PositionalEncodingTF(d_pe, max_len, MAX)
        self.pos_encoder_sensor = PositionalEncodingTF(d_pe, max_len, MAX)

        self.linear_value = nn.Linear(1, 16)
        self.linear_sensor = nn.Linear(1, 16)

        self.d_K = 2 * (d_pe+ 16+16)

        encoder_layers = TransformerEncoderLayer(self.d_K, 1, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        encoder_layers_f_prime = TransformerEncoderLayer(int(self.d_K//2), 1, nhid, dropout)
        self.transformer_encoder_f_prime = TransformerEncoder(encoder_layers_f_prime, 2)

        if static:
            self.emb = nn.Linear(d_static, 16)

        self.proj_weight = Parameter(torch.Tensor(self.d_K, 128))

        self.lin_map = nn.Linear(self.d_K, 128)

        d_fi = 128 + 16

        if static == False:
            d_fi = 128
        else:
            d_fi = 128 + d_pe

        if configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            self.mlp = nn.Sequential(
                nn.Linear(d_fi, d_fi),
                nn.ReLU(),
                nn.Linear(d_fi, configs.pred_len_max_irr or configs.pred_len),
            )
        elif configs.task_name == "classification":
            self.decoder_classification = nn.Sequential(
                nn.Linear(d_inp * d_fi, d_inp * d_fi),
                nn.ReLU(),
                nn.Linear(d_inp * d_fi, configs.n_classes),
            )
        else:
            raise NotImplementedError

        self.aggreg = aggreg

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        self.linear_value.weight.data.uniform_(-initrange, initrange)
        self.linear_sensor.weight.data.uniform_(-initrange, initrange)
        self.lin_map.weight.data.uniform_(-initrange, initrange)
        xavier_uniform_(self.proj_weight)

    def forward(self,
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
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

        src = torch.cat([x, x_mask], dim=-1).permute(1, 0, 2)
        static = None
        times = x_mark[:, :, :1].permute(1, 0, 2)
        # END adaptor

        maxlen, batch_size = src.shape[0], src.shape[1]

        src = src.permute(1, 0, 2)
        fea = src[:, :, :int(src.shape[2]/2)]

        output = torch.zeros((batch_size, ENC_IN, self.d_K)).to(x.device)
        for i in range(batch_size):
            nonzero_index = fea[i].nonzero(as_tuple=False)
            if nonzero_index.shape[0] == 0:
                continue
            values = fea[i][nonzero_index[:,0], nonzero_index[:,1]] # v in SEFT paper N_OBSERVATIONS_MAX
            time_index = nonzero_index[:, 0]
            time_sequence = times[:, i]
            time_points = time_sequence[time_index]  # t in SEFT paper
            pe_ = self.pos_encoder(time_points).squeeze(1)

            variable = nonzero_index[:, 1]  # the dimensions of variables. The m value in SEFT paper.

            unit = torch.cat([pe_, values.unsqueeze(1), variable.unsqueeze(1)], dim=1)

            variable_ = self.pos_encoder_sensor(variable.unsqueeze(1)).squeeze(1)

            values_ = self.linear_value(values.float().unsqueeze(1)).squeeze(1)

            unit = torch.cat([pe_, values_, variable_], dim=1)

            f_prime = torch.mean(unit, dim=0)

            x = torch.cat([f_prime.repeat(unit.shape[0], 1), unit], dim=1)

            output_unit = x 
            output_unit = self.unpad_and_reshape(
                tensor_flattened=output_unit,
                original_mask=fea[i]
            ) # (N_OBSERVATIONS_MAX, d_K) -> (SEQ_LEN, ENC_IN, d_K)

            # set padding values to nan for ease of nanmean calculation
            output_unit[repeat(fea[i], "L N -> L N d_K", d_K=self.d_K) == 0] = torch.nan
            output_unit = torch.nanmean(output_unit, dim=0) # TODO: unpad then perform nan mean along each variable
            output_unit = output_unit.nan_to_num()
            output[i] = output_unit

        output = self.lin_map(output)

        if static is not None:
            emb = self.emb(static)

        # feed through MLP
        if static is not None:
            output = torch.cat([output, emb], dim=1)

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            output = rearrange(
                self.mlp(output),
                "B N L -> B L N"
            )
            return {
                "pred": output[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.decoder_classification(output.view(BATCH_SIZE, -1)),
                "true_class": y_class
            }
        else:
            raise NotImplementedError

    # convert the output back to original shape, to align with api
    def unpad_and_reshape(
        self, 
        tensor_flattened: Tensor, 
        original_mask: Tensor, 
    ):
        N_OBSERVATIONS_MAX, hidden_dim = tensor_flattened.shape
        L, ENC_IN = original_mask.shape
        result = torch.zeros((L, ENC_IN, hidden_dim), dtype=tensor_flattened.dtype, device=tensor_flattened.device)

        masked_indices = repeat(
            original_mask.reshape(-1).nonzero(as_tuple=True)[0],
            "N_OBSERVATIONS_MAX -> N_OBSERVATIONS_MAX hidden_dim",
            hidden_dim=hidden_dim
        )
        unpadded_sequence = tensor_flattened[:len(masked_indices)]
        result.view(-1)[masked_indices] = unpadded_sequence
            
        return result

class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time).to(P_time.device)
        return pe