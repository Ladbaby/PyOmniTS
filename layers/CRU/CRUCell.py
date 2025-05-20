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

from typing import Iterable, Tuple, List, Union

import torch
import torch.nn as nn
import numpy as np

from utils.ExpConfigs import ExpConfigs


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def bmv(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Batched Matrix Vector Product"""
    return torch.bmm(mat, vec[..., None])[..., 0]


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def dadat(a: torch.Tensor, diag_mat: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * A^T) where A is a batch of square matrices and
    diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch of square matrices,
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :returns diagonal entries of  A * diag_mat * A^T"""
    return bmv(a.square(), diag_mat)


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def dadbt(a: torch.Tensor, diag_mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched computation of diagonal entries of (A * diag_mat * B^T) where A and B are batches of square matrices and
     diag_mat is a batch of diagonal matrices (represented as vectors containing diagonal entries)
    :param a: batch square matrices
    :param diag_mat: batch of diagonal matrices (represented as vecotrs containing diagonal entries
    :param b: batch of square matrices
    :returns diagonal entries of  A * diag_mat * B^T"""
    return bmv(a * b, diag_mat)


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def dadb(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, -1)


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def var_activation(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation function to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.log(torch.exp(x) + 1.0)


# taken from https://github.com/ALRhub/rkn_share/ and not modified
def var_activation_inverse(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(np.exp(x) - 1.0)


# taken from https://github.com/ALRhub/rkn_share/ and modified
class RKNCell(nn.Module):

# taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(
        self, 
        latent_obs_dim: int, 
        configs: ExpConfigs, 
        dtype: torch.dtype = torch.float32
    ):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param args: args object, for configuring the cell
        :param dtype: dtype for input data
        """
        super(RKNCell, self).__init__()
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod
        self.configs = configs

        self._dtype = dtype

        self.trans_var_activation = "relu"
        self.trans_covar = 0.1
        self.t_sensitive_trans_net = False
        self.f_cru = False
        self.orthogonal = True
        self.rkn = False
        self.trans_net_hidden_units = []
        self.trans_net_hidden_activation = "tanh"

        self._build_transition_model()

    # new code component
    def _var_activation(self):
        if self.trans_var_activation == 'exp':
            return torch.exp(self._log_transition_noise)
        elif self.trans_var_activation == 'relu':
            return torch.maximum(self._log_transition_noise, torch.zeros_like(self._log_transition_noise))
        elif self.trans_var_activation == 'square':
            return torch.square(self._log_transition_noise)
        elif self.trans_var_activation == 'abs':
            return torch.abs(self._log_transition_noise)
        else:
            return torch.log(torch.exp(self._log_transition_noise) + 1.0)

    # new code component
    def _var_activation_inverse(self):
        if self.trans_var_activation == 'exp':
            return np.log(self.trans_covar)
        elif self.trans_var_activation == 'relu':
            return self.trans_covar
        elif self.trans_var_activation == 'square':
            return np.sqrt(self.trans_covar)
        elif self.trans_var_activation == 'abs':
            return self.trans_covar
        else:
            return np.log(np.exp(self.trans_covar) - 1.0)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _compute_band_util(self, lod: int, bandwidth: int):
        self._num_entries = lod + 2 * np.sum(np.arange(lod - bandwidth, lod))
        np_mask = np.ones([lod, lod], dtype=np.float64)
        np_mask = np.triu(np_mask, -bandwidth) * np.tril(np_mask, bandwidth)
        mask = torch.tensor(np_mask, dtype=torch.bool)
        idx = torch.where(mask == 1)
        diag_idx = torch.where(idx[0] == idx[1])

        self.register_buffer("_idx0", idx[0], persistent=False)
        self.register_buffer("_idx1", idx[1], persistent=False)
        self.register_buffer("_diag_idx", diag_idx[0], persistent=False)

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _unflatten_tm(self, tm_flat: torch.Tensor) -> torch.Tensor:
        tm = torch.zeros(
            tm_flat.shape[0], self._lod, self._lod, device=tm_flat.device, dtype=self._dtype)
        tm[:, self._idx0, self._idx1] = tm_flat
        return tm

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor = None, delta_t: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Iterable[torch.Tensor], torch.Tensor, Iterable[torch.Tensor]]:
        """Forward pass trough the cell. 

        :param prior_mean: prior mean at time t
        :param prior_cov: prior covariance at time t
        :param obs: observation at time t
        :param obs_var: observation variance at time t
        :param obs_valid: boolean indicating whether observation at time t is valid
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :return: posterior mean at time t, posterior covariance at time t
                 prior mean at time t', prior covariance at time t', Kalman gain at time t
        """

        post_mean, post_cov, kalman_gain = self._update(
            prior_mean, prior_cov, obs, obs_var, obs_valid)

        next_prior_mean, next_prior_covar = self._predict(
            post_mean, post_cov, delta_t=delta_t)

        return post_mean, post_cov, next_prior_mean, next_prior_covar, kalman_gain

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _build_coefficient_net(self, num_hidden: Iterable[int], activation: str) -> torch.nn.Sequential:
        """Builds the network computing the coefficients from the posterior mean. Currently only fully connected
        neural networks with same activation across all hidden layers supported
        :param num_hidden: number of hidden units per layer
        :param activation: hidden activation
        :return: coefficient network
        """
        layers = []
        prev_dim = self._lsd + 1 if self.t_sensitive_trans_net else self._lsd
        for n in num_hidden:
            layers.append(nn.Linear(prev_dim, n))
            layers.append(getattr(nn, activation)())
            prev_dim = n
        layers.append(nn.Linear(prev_dim, self.configs.cru_num_basis))
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers).to(dtype=self._dtype)

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _build_transition_model(self) -> None:
        """
        Builds the basis functions for transition model and the noise
        :return:
        """

        # initialize eigenvectors E and eigenvalues d of state transition matrix
        if self.f_cru:
            self.E = nn.Linear(self._lsd, self._lsd, bias=False).double()
            self.d = nn.Parameter(
                1e-5 + torch.zeros(self.configs.cru_num_basis, self._lsd, dtype=torch.float32))

            if self.orthogonal:
                import geotorch
                geotorch.orthogonal(self.E, 'weight')
                self.E.weight = torch.eye(
                    self._lsd, self._lsd, dtype=torch.float32)

        else:
            # build state independent basis matrices
            self._compute_band_util(lod=self._lod, bandwidth=self.configs.cru_bandwidth)
            self._tm_11_basis = nn.Parameter(torch.zeros(
                self.configs.cru_num_basis, self._num_entries, dtype=self._dtype))
            
            tm_12_init = torch.zeros(
                self.configs.cru_num_basis, self._num_entries, dtype=self._dtype)
            if self.rkn:
                tm_12_init[:, self._diag_idx] += 0.2 * torch.ones(self._lod)
            self._tm_12_basis = nn.Parameter(tm_12_init)
            
            tm_21_init = torch.zeros(
                self.configs.cru_num_basis, self._num_entries, dtype=self._dtype)
            if self.rkn:
                tm_21_init[:, self._diag_idx] -= 0.2 * torch.ones(self._lod)
            self._tm_21_basis = nn.Parameter(tm_21_init)

            self._tm_22_basis = nn.Parameter(torch.zeros(
                self.configs.cru_num_basis, self._num_entries, dtype=self._dtype))

            self._transition_matrices_raw = [
                self._tm_11_basis, self._tm_12_basis, self._tm_21_basis, self._tm_22_basis]

        self._coefficient_net = self._build_coefficient_net(self.trans_net_hidden_units, self.trans_net_hidden_activation)

        init_log_trans_cov = self._var_activation_inverse()
        self._log_transition_noise = \
            nn.Parameter(nn.init.constant_(torch.empty(
                1, self._lsd, dtype=self._dtype), init_log_trans_cov))

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def get_transition_model(self, post_mean: torch.Tensor, delta_t: torch.Tensor = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute the locally-linear transition model given the current posterior mean
        :param post_mean: current posterior mean
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :return: transition matrices for CRU and RKN or their eigenvectors and eigenvalues for f-CRU.
        """
        trans_net_input = torch.cat(
            [post_mean, delta_t[:, None]], 1) if self.t_sensitive_trans_net else post_mean
        coefficients = torch.reshape(self._coefficient_net(trans_net_input), [-1, self.configs.cru_num_basis, 1])  # [batchsize, c.num_basis, 1]
        
        if self.f_cru:
            # coefficients (batchsize, K, 1)
            eigenvalues = (coefficients * self.d).sum(dim=1)
            eigenvectors = self.E.weight
            transition = [eigenvalues, eigenvectors]

        else:
            # [batchsize, 93]
            tm11_flat = (coefficients * self._tm_11_basis).sum(dim=1)
            tm12_flat = (coefficients * self._tm_12_basis).sum(dim=1)
            tm21_flat = (coefficients * self._tm_21_basis).sum(dim=1)
            tm22_flat = (coefficients * self._tm_22_basis).sum(dim=1)

            # impose diagonal structure for transition matrix of RKN 
            if self.rkn:
                tm11_flat[:, self._diag_idx] += 1.0
                tm22_flat[:, self._diag_idx] += 1.0

            transition = [self._unflatten_tm(
                x) for x in [tm11_flat, tm12_flat, tm21_flat, tm22_flat]]

        trans_cov = self._var_activation()

        return transition, trans_cov

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _update(self, prior_mean: torch.Tensor, prior_cov: Iterable[torch.Tensor],
                obs_mean: torch.Tensor, obs_var: torch.Tensor, obs_valid: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Performs update step
        :param prior_mean: current prior state mean
        :param prior_cov: current prior state covariance
        :param obs_mean: current observation mean
        :param obs_var: current covariance mean
        :param obs_valid: flag if current time point is valid
        :return: current posterior state and covariance
        """
        cov_u, cov_l, cov_s = prior_cov

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = cov_u + obs_var
        q_upper = cov_u / denominator
        q_lower = cov_s / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + \
            torch.cat([q_upper * residual, q_lower * residual], -1)

        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * cov_u
        new_covar_lower = cov_l - q_lower * cov_s
        new_covar_side = covar_factor * cov_s

        # ensures update only happens if an observation is given, otherwise posterior is set to prior
        obs_valid = obs_valid[..., None]
        masked_mean = new_mean.where(obs_valid, prior_mean)
        masked_covar_upper = new_covar_upper.where(obs_valid, cov_u)
        masked_covar_lower = new_covar_lower.where(obs_valid, cov_l)
        masked_covar_side = new_covar_side.where(obs_valid, cov_s)

        return masked_mean, [masked_covar_upper, masked_covar_lower, masked_covar_side], [q_upper, q_lower]


    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], delta_t: torch.Tensor = None) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs prediction step for regular time intervals (RKN variant)
        :param post_mean: last posterior mean
        :param post_covar: last posterior covariance
        :param delta_t: ignored for discrete RKN
        :return: current prior state mean and covariance
        """
        # compute state dependent transition matrix
        [tm11, tm12, tm21, tm22], trans_covar = self.get_transition_model(
            post_mean, delta_t)

        # prepare transition noise
        trans_covar_upper = trans_covar[..., :self._lod]
        trans_covar_lower = trans_covar[..., self._lod:]

        # predict next prior mean
        mu = post_mean[:, :self._lod]
        ml = post_mean[:, self._lod:]

        nmu = bmv(tm11, mu) + bmv(tm12, ml)
        nml = bmv(tm21, mu) + bmv(tm22, ml)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        cu, cl, cs = post_covar
        ncu = dadat(tm11, cu) + 2.0 * dadbt(tm11, cs, tm12) + \
            dadat(tm12, cl) + trans_covar_upper
        ncl = dadat(tm21, cu) + 2.0 * dadbt(tm21, cs, tm22) + \
            dadat(tm22, cl) + trans_covar_lower
        ncs = dadbt(tm21, cu, tm11) + dadbt(tm22, cs, tm11) + \
            dadbt(tm21, cs, tm12) + dadbt(tm22, cl, tm12)
        return torch.cat([nmu, nml], dim=-1), [ncu, ncl, ncs]



# Continuous Discrete Kalman Cell: the prediction function assumes continuous states
# new code component 
class CRUCell(RKNCell):
    def __init__(self, latent_obs_dim: int, configs, dtype: torch.dtype = torch.float32):
        super(CRUCell, self).__init__(latent_obs_dim, configs, dtype)

    def get_prior_covar_vanloan(self, post_covar, delta_t, Q, A, exp_A):
        """Computes Prior covariance matrix based on matrix fraction decomposition proposed by Van Loan.
        See Appendix A.2.1 in paper. This function is used for CRU.

        :param post_covar: posterior covariance at time t
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :param Q: diffusion matrix of Brownian motion in SDE that governs state evolution
        :param A: transition matrix
        :param exp_A: matrix exponential of (A * delta_t)
        :return: prior covariance at time t'
        """

        h2 = Q
        h3 = torch.zeros_like(h2)
        h4 = (-1) * torch.transpose(A, -2, -1)
        assert h2.shape == h3.shape == h4.shape , 'shapes must be equal (batchsize, latent_state_dim, latent_state_dim)'
        
        # construct matrix B (eq 27 in paper)
        B = torch.cat((torch.cat((A, h2), -1), torch.cat((h3, h4), -1)), -2) # (batchsize, 2*latent_state_dim, 2*latent_state_dim)
        exp_B = torch.matrix_exp(B * delta_t)
        M1 = exp_B[:, :self._lsd, :self._lsd] # (batchsize, latent_state_dim, latent_state_dim)
        M2 = exp_B[:, :self._lsd, self._lsd:]

        if torch.all(torch.isclose(M1, exp_A, atol=1e-8)) == False:
            print('---- ASSERTION M1 and exp_A are not identical ----')

        # compute prior covar (eq 28 in paper)
        C = torch.matmul(exp_A, post_covar) + M2
        prior_covar = torch.matmul(C, torch.transpose(exp_A, -2, -1))

        # for tensorboard plotting
        self.exp_B = exp_B
        self.M2 = M2

        return prior_covar

    def get_prior_covar_rome(self, post_covar, delta_t, Q, d, eigenvectors):
        """Computes prior covariance matrix based on the eigendecomposition of the transition matrix proposed by
        Rome (1969) https://ieeexplore.ieee.org/document/1099271. This function is used for f-CRU.

        :param post_covar: posterior covariance at time t
        :param delta_t: time interval between current observation at time t and next observation at time t'
        :param Q: diffusion matrix of Brownian motion in SDE that governs state evolution
        :param d: eigenvalues of transition matrix
        :param eigenvectors: eigenvectors of transition matrix
        :return: prior covariance at time t'
        """

        jitter = 0  # 1e-8
        if self.orthogonal:
            eigenvectors_inverse = torch.transpose(eigenvectors, -2, -1)
            eigenvectors_inverse_trans = eigenvectors
        else:
            # not used in paper (no speed up, unstable)
            eigenvectors_inverse = torch.inverse(eigenvectors)
            eigenvectors_inverse_trans = torch.transpose(
                torch.inverse(eigenvectors), -1, -2)

        # compute Sigma_w of current time step (eq 22 in paper)
        Sigma_w = torch.matmul(eigenvectors_inverse, torch.matmul(
            post_covar, eigenvectors_inverse_trans))

        # compute D_tilde with broadcasting (eq 23 in paper)
        D_tilde = d[:, :, None] + d[:, None, :]
        exp_D_tilde = torch.exp(D_tilde * delta_t)

        # compute S (eq 24 in paper)
        S = torch.matmul(eigenvectors_inverse, torch.matmul(
            Q, eigenvectors_inverse_trans))

        # compute Sigma_w of next time step with elementwise multiplication/division (eq 25 in paper)
        Sigma_w_next = (S * exp_D_tilde - S) / (D_tilde + jitter) + Sigma_w * exp_D_tilde

        # compute prior_covar (eq 26 in paper)
        prior_covar = torch.matmul(eigenvectors, torch.matmul(
            Sigma_w_next, torch.transpose(eigenvectors, -1, -2)))

        return prior_covar


    def _predict(self, post_mean: torch.Tensor, post_covar: List[torch.Tensor], delta_t: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """ Performs continuous prediction step for irregularly sampled data 
        :param post_mean: last posterior mean
        :param post_cov: last posterior covariance
        :param delta_t: time delta to next observation
        :return: next prior state mean and covariance
        """

        delta_t = delta_t[:, None, None] if delta_t is not None else 1
        transition, trans_covar = self.get_transition_model(post_mean, delta_t)
        trans_covar = torch.diag_embed(
            trans_covar.repeat(post_mean.shape[0], 1))

        # build full covariance matrix
        post_cu, post_cl, post_cs = [torch.diag_embed(x) for x in post_covar]
        post_covar = torch.cat(
            (torch.cat((post_cu, post_cs), -1), torch.cat((post_cs, post_cl), -1)), -2)

        if self.f_cru:
            eigenvalues, eigenvectors = transition

            # compute prior mean (eq 21 in paper)
            exp_D = torch.diag_embed(
                torch.exp(eigenvalues * delta_t.squeeze(-1)))
            eigenvectors_inverse = torch.transpose(
                eigenvectors, -2, -1) if self.orthogonal else torch.inverse(eigenvectors)
            exp_A = torch.matmul(eigenvectors, torch.matmul(
                exp_D, eigenvectors_inverse))
            prior_mean = bmv(exp_A, post_mean)

            # compute prior covariance (eq 22 - 26 in paper)
            prior_covar = self.get_prior_covar_rome(
                    post_covar, delta_t, trans_covar, eigenvalues, eigenvectors)
        
        else: 
            [tm11, tm12, tm21, tm22] = transition
            A = torch.cat((torch.cat((tm11, tm12), -1),
                          torch.cat((tm21, tm22), -1)), -2)
            exp_A = torch.matrix_exp(A * delta_t)

            prior_mean = bmv(exp_A, post_mean)
            prior_covar = self.get_prior_covar_vanloan(
                post_covar, delta_t, trans_covar, A, exp_A)
            self.A = A

        # extract diagonal elements of prior covariance
        ncu = torch.diagonal(
            prior_covar[:, :self._lod, :self._lod], dim1=-1, dim2=-2)
        ncl = torch.diagonal(
            prior_covar[:, self._lod:, self._lod:], dim1=-1, dim2=-2)
        ncs = torch.diagonal(
            prior_covar[:, :self._lod, self._lod:], dim1=-1, dim2=-2)
        ncs2 = torch.diagonal(
            prior_covar[:, self._lod:, :self._lod], dim1=-1, dim2=-2)

        if torch.all(torch.isclose(ncs, ncs2, atol=1e-2)) == False:
            print('---- ASSERTION ncs not identical ----')

        # for tensorboard plotting
        self.exp_A = exp_A
        self.trans_covar = trans_covar

        return prior_mean, [ncu, ncl, ncs]
