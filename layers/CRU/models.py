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

from lib.CRU import CRU
from lib.utils import MyLayerNorm2d
import torch
nn = torch.nn


# new code component
def load_model(configs):
    
    # Pendulum 
    if configs.dataset == 'pendulum':
        if configs.task =='regression':
            model = Pendulum_reg(target_dim=2, lsd=configs.latent_state_dim, configs=configs, 
                layer_norm=False, use_cuda_if_available=True)
        elif configs.task == 'interpolation':
            model = Pendulum(target_dim=(1, 24, 24), lsd=configs.latent_state_dim, configs=configs, 
                layer_norm=True, use_cuda_if_available=True, bernoulli_output=True)
        else:
            raise Exception('Task not available for Pendulum data')
        
    # USHCN
    elif configs.dataset == 'ushcn':
        model = Physionet_USHCN(target_dim=5, lsd=configs.latent_state_dim, configs=configs,
                use_cuda_if_available=True)

    # Physionet
    elif configs.dataset == 'physionet':
        model = Physionet_USHCN(target_dim=37, lsd=configs.latent_state_dim, configs=configs,
                use_cuda_if_available=True)

    return model


# new code component
class Physionet_USHCN(CRU):

    def __init__(self, target_dim: int, lsd: int, configs,
                 use_cuda_if_available: bool = True):
        self.hidden_units = configs.hidden_units
        self.target_dim = target_dim

        super(Physionet_USHCN, self).__init__(target_dim, lsd,
                                             configs, use_cuda_if_available)


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
        return nn.ModuleList(layers).to(dtype=torch.float32), self.hidden_units

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
        ]).to(dtype=torch.float32), self.hidden_units

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=self.hidden_units),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_units)
        ]).to(dtype=torch.float32), self.hidden_units


# taken from https://github.com/ALRhub/rkn_share/ and modified
class Pendulum(CRU):

    def __init__(self, target_dim: int, lsd: int, configs, layer_norm: bool,
                 use_cuda_if_available: bool = True, bernoulli_output=True):

        self._layer_norm = layer_norm
        super(Pendulum, self).__init__(target_dim, lsd, configs, use_cuda_if_available, bernoulli_output)

    def _build_enc_hidden_layers(self):
        layers = []
        # hidden layer 1
        layers.append(nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 2
        layers.append(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=2, padding=1))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 3
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=108, out_features=30))
        layers.append(nn.ReLU())
        return nn.ModuleList(layers), 30

    def _build_dec_hidden_layers(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=144),
            nn.ReLU(),
            nn.Unflatten(1, [16, 3, 3]),

            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=4, padding=2),
            MyLayerNorm2d(channels=16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=3, stride=2, padding=1),
            MyLayerNorm2d(channels=12),
            nn.ReLU()
        ]), 12


# taken from https://github.com/ALRhub/rkn_share/ and modified
class Pendulum_reg(CRU):

    def __init__(self, target_dim: int, lsd: int, configs, layer_norm: bool,
                 use_cuda_if_available: bool = True):

        self._layer_norm = layer_norm
        super(Pendulum_reg, self).__init__(target_dim, lsd, configs, use_cuda_if_available)

    def _build_enc_hidden_layers(self):
        layers = []
        # hidden layer 1
        layers.append(nn.Conv2d(in_channels=1, out_channels=12,
                      kernel_size=5, padding=2))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 2
        layers.append(nn.Conv2d(in_channels=12, out_channels=12,
                      kernel_size=3, stride=2, padding=1))
        if self._layer_norm:
            layers.append(MyLayerNorm2d(channels=12))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # hidden layer 3
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=108, out_features=30))
        layers.append(nn.ReLU())
        return nn.ModuleList(layers).to(dtype=torch.float32), 30

    def _build_dec_hidden_layers_mean(self):
        return nn.ModuleList([
            nn.Linear(in_features=2 * self._lod, out_features=30),
            nn.Tanh()
        ]), 30

    def _build_dec_hidden_layers_var(self):
        return nn.ModuleList([
            nn.Linear(in_features=3 * self._lod, out_features=30),
            nn.Tanh()
        ]), 30




