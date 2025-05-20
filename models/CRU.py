# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from einops import *

from layers.CRU.CRU import CRU
from layers.CRU.CRUCell import var_activation
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(CRU):
    '''
    - paper: "Modeling Irregular Time Series with Continuous Recurrent Units" (ICML 2022)
    - paper link: https://proceedings.mlr.press/v162/schirmer22a.html
    - code adapted from: https://github.com/boschresearch/Continuous-Recurrent-Units
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
    ):
        self.hidden_units = 50
        self.target_dim = configs.enc_in 
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        super(Model, self).__init__(
            configs.enc_in, 
            configs.d_model, # lsd
            configs, 
            True 
        )

        if configs.task_name == "classification":
            logger.exception("AutoformerMS (Scaleformer) does not support classification task!", stack_info=True)
            exit(1)

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None,
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
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        x_padding = torch.zeros_like(y)
        x = torch.cat([x, x_padding], dim=1)
        x_mark = torch.cat([x_mark, y_mark], dim=1)[:, :, 0]
        x_mask = torch.cat([x_mask, y_mask], dim=1)[:, :, 0].bool()
        # END adaptor

        x_enc, x_var = self._enc(x)
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(
            x_enc, 
            x_var, 
            self._initial_mean.to(x.device),
            [var_activation(self._log_icu.to(x.device)), var_activation(self._log_icl.to(x.device)), self._ics.to(x.device)],
            obs_valid=x_mask, 
            time_points=x_mark
        )
        if self.bernoulli_output:
            # output an image
            out_mean = self._dec(post_mean)
            out_var = None
        elif self.configs.task_name == 'one_step_ahead_prediction':
            # output prediction for the next time step
            out_mean, out_var = self._dec(
                prior_mean, torch.cat(prior_cov, dim=-1))
        else:
            # output filtered observation
            out_mean, out_var = self._dec(
                post_mean, torch.cat(post_cov, dim=-1))

        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            PRED_LEN = y.shape[1]
            f_dim = -1 if self.configs.features == 'MS' else 0
            return {
                "pred": out_mean[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError


    def _build_enc_hidden_layers(self):
        layers = []
        layers.append(nn.Linear(self.target_dim, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))

        layers.append(nn.Linear(self.hidden_units, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))

        layers.append(nn.Linear(self.hidden_units, self.hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(self.hidden_units))
        # size last hidden
        return nn.ModuleList(layers).to(dtype=torch.float64), self.hidden_units

    def _build_dec_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units),

            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units),

            nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units)
        ]).to(dtype=torch.float64), self.hidden_units

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units)
        ]).to(dtype=torch.float64), self.hidden_units