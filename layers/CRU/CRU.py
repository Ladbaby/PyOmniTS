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

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from layers.CRU.utils import TimeDistributed
from layers.CRU.encoder import Encoder
from layers.CRU.decoder import SplitDiagGaussianDecoder, BernoulliDecoder
from layers.CRU.CRULayer import CRULayer
from layers.CRU.CRUCell import var_activation, var_activation_inverse
from layers.CRU.losses import rmse, mse, GaussianNegLogLik, bernoulli_nll
from layers.CRU.data_utils import  align_output_and_target, adjust_obs_for_extrapolation
from utils.ExpConfigs import ExpConfigs

# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRU(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(
        self, 
        target_dim: int, 
        lsd: int, 
        configs: ExpConfigs, 
        use_cuda_if_available: bool = True, 
        bernoulli_output: bool = False
    ):
        """
        :param target_dim: output dimension
        :param lsd: latent state dimension
        :param args: parsed arguments
        :param use_cuda_if_available: if to use cuda or cpu
        :param use_bernoulli_output: if to use a convolutional decoder (for image data)
        """
        super().__init__()

        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2) 
        else:
            raise Exception('Latent state dimension must be even number.')
        self.configs = configs

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = self.configs.learning_rate
        self.bernoulli_output = bernoulli_output
        # main model

        self._cru_layer = CRULayer(
            latent_obs_dim=self._lod, configs=configs)

        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(self._lod, output_normalization=self._enc_out_normalization,
                      enc_var_activation="square").to(dtype=torch.float32)

        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=target_dim, configs=configs).to(torch.float32), num_outputs=1, low_mem=True)
            self._enc = TimeDistributed(
                enc, num_outputs=2, low_mem=True)

        else:
            SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
            SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
            self._dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=target_dim, dec_var_activation="exp").to(
                dtype=torch.float32), num_outputs=2)
            self._enc = TimeDistributed(enc, num_outputs=2)

        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(torch.float32)
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(torch.float32))
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(torch.float32))
        self._ics = torch.zeros(1, self._lod).to(
            torch.float32)

        # params and optimizer
        self._params = list(self._enc.parameters())
        self._params += list(self._cru_layer.parameters())
        self._params += list(self._dec.parameters())
        self._params += [self._log_icu, self._log_icl]

        self._optimizer = optim.Adam(self._params, lr=self.configs.learning_rate)
        self._shuffle_rng = np.random.RandomState(
            42)  # rng for shuffling batches

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError
    
    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(
        self, 
        obs_batch: torch.Tensor, 
        time_points: torch.Tensor = None, 
        obs_valid: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass on a batch
        :param obs_batch: batch of observation sequences
        :param time_points: timestamps of observations
        :param obs_valid: boolean if timestamp contains valid observation 
        """
        y, y_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(
            y, 
            y_var, 
            self._initial_mean,
            [var_activation(self._log_icu), var_activation(self._log_icl), self._ics],
            obs_valid=obs_valid, 
            time_points=time_points
        )
        # output an image
        if self.bernoulli_output:
            out_mean = self._dec(post_mean)
            out_var = None

        # output prediction for the next time step
        elif self.configs.task_name == 'one_step_ahead_prediction':
            out_mean, out_var = self._dec(
                prior_mean, torch.cat(prior_cov, dim=-1))

        # output filtered observation
        else:
            out_mean, out_var = self._dec(
                post_mean, torch.cat(post_cov, dim=-1))

        intermediates = {
            'post_mean': post_mean,
            'post_cov': post_cov,
            'prior_mean': prior_mean,
            'prior_cov': prior_cov,
            'kalman_gain': kalman_gain,
            'y': y,
            'y_var': y_var
        }

        return out_mean, out_var, intermediates

    # new code component
    def interpolation(self, data, track_gradient=True):
        """Computes loss on interpolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        if self.bernoulli_output:
            obs, truth, obs_valid, obs_times, mask_truth = [
                j for j in data]
            mask_obs = None
        else:
            obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
                j for j in data]

        obs_times = self.configs.cru_ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)

            if self.bernoulli_output:
                loss = bernoulli_nll(truth, output_mean, uint8_targets=False)
                mask_imput = (~obs_valid[...,None, None, None]) * mask_truth
                imput_loss = np.nan #TODO: compute bernoulli loss on imputed points
                imput_mse = mse(truth.flatten(start_dim=2), output_mean.flatten(start_dim=2), mask=mask_imput.flatten(start_dim=2))

            else:
                loss = GaussianNegLogLik(
                    output_mean, truth, output_var, mask=mask_truth)
                # compute metric on imputed points only
                mask_imput = (~obs_valid[...,None]) * mask_truth
                imput_loss = GaussianNegLogLik(output_mean, truth, output_var, mask=mask_imput)
                imput_mse = mse(truth, output_mean, mask=mask_imput)
        
        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    # new code component
    def extrapolation(self, data, track_gradient=True):
        """Computes loss on extrapolation task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, outputs, inputs, intermediate variables, metrics on imputed points
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j for j in data]
        obs, obs_valid = adjust_obs_for_extrapolation(
            obs, obs_valid, obs_times, self.configs.cut_time)
        obs_times = self.configs.cru_ts * obs_times

        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)

            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)
            
            # compute metric on imputed points only
            mask_imput = (~obs_valid[..., None]) * mask_truth
            imput_loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_imput)
            imput_mse = mse(truth, output_mean, mask=mask_imput)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates, imput_loss, imput_mse

    # new code component
    def regression(self, data, track_gradient=True):
        """Computes loss on regression task

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, input, intermediate variables and computed output
        """
        obs, truth, obs_times, obs_valid = [j for j in data]
        mask_truth = None
        mask_obs = None
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates

    # new code component
    def one_step_ahead_prediction(self, data, track_gradient=True):
        """Computes loss on one-step-ahead prediction

        :param data: batch of data
        :param track_gradient: if to track gradient for backpropagation
        :return: loss, input, intermediate variables and computed output
        """
        obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [
            j for j in data]
        obs_times = self.configs.cru_ts * obs_times
        with torch.set_grad_enabled(track_gradient):
            output_mean, output_var, intermediates = self.forward(
                obs_batch=obs, time_points=obs_times, obs_valid=obs_valid)
            output_mean, output_var, truth, mask_truth = align_output_and_target(
                output_mean, output_var, truth, mask_truth)
            loss = GaussianNegLogLik(
                output_mean, truth, output_var, mask=mask_truth)

        return loss, output_mean, output_var, obs, truth, mask_obs, mask_truth, intermediates

