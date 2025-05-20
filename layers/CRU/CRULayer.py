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

from layers.CRU.CRUCell import RKNCell, CRUCell


# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRULayer(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, latent_obs_dim, configs, dtype=torch.float32):
        super().__init__()
        self.rkn = False

        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._cell = RKNCell(latent_obs_dim, configs, dtype) if self.rkn else CRUCell(latent_obs_dim, configs, dtype)

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, latent_obs, obs_vars, initial_mean, initial_cov, obs_valid=None, time_points=None):
        """Passes the entire observation sequence sequentially through the Kalman component

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating if observation is valid 
        :param time_points: timestamp of the observation
        """

        # prepare list for return
        prior_mean_list = []
        prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]
        kalman_gain_list = [[], []]

        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov
        T = latent_obs.shape[1]

        # iterate over sequence length
        for i in range(T):
            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
            delta_t = time_points[:, i+1] - time_points[:,
                                                        i] if time_points is not None and i < T-1 else torch.ones_like(latent_obs)[:, 0, 0]
            post_mean, post_cov, next_prior_mean, next_prior_cov, kalman_gain = \
                self._cell(prior_mean, prior_cov,
                           latent_obs[:, i], obs_vars[:, i], cur_obs_valid, delta_t=delta_t)
            #print(f'post_mean {post_mean.shape}, next_prior_mean {next_prior_mean.shape}')

            post_mean_list.append(post_mean)
            [post_cov_list[i].append(post_cov[i]) for i in range(3)]
            prior_mean_list.append(next_prior_mean)
            [prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]
            [kalman_gain_list[i].append(kalman_gain[i]) for i in range(2)]

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = [torch.stack(x, 1) for x in prior_cov_list]
        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]
        kalman_gains = [torch.stack(x, 1) for x in kalman_gain_list]

        return post_means, post_covs, prior_means, prior_covs, kalman_gains
