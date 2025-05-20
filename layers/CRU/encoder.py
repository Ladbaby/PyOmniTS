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

import torch
import torch.nn as nn

# taken from https://github.com/ALRhub/rkn_share/ and not modified
def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


# new code component
def var_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


# taken from https://github.com/ALRhub/rkn_share/ and modified
class Encoder(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, lod: int, enc_var_activation: str, output_normalization: str = "post"):
        """Gaussian Encoder, as described in RKN ICML Paper (if output_normalization=post)
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param enc_var_activation: activation function for latent observation noise
        :param output_normalization: when to normalize the output:
            - post: after output layer 
            - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
            - none: (or any other string) not at all

        """
        super(Encoder, self).__init__()
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                               "not found by the optimizer"
        self._mean_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(
            in_features=size_last_hidden, out_features=lod)
        self.enc_var_activation = enc_var_activation
        self._output_normalization = output_normalization

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = obs
        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        log_var = self._log_var_layer(h)

        if self.enc_var_activation == 'exp':
            var = torch.exp(log_var)
        elif self.enc_var_activation == 'relu':
            var = torch.maximum(log_var, torch.zeros_like(log_var))
        elif self.enc_var_activation == 'square':
            var = torch.square(log_var)
        elif self.enc_var_activation == 'abs':
            var = torch.abs(log_var)
        elif self.enc_var_activation == 'elup1':
            var = torch.exp(log_var).where(log_var < 0.0, log_var + 1.0)
        else: 
            raise Exception('Variance activation function unknown.')
        return mean, var
