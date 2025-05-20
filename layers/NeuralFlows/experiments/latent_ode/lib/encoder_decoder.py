import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *

import layers.Latent_ODE.utils as utils

def get_mask(x):
    x = x.unsqueeze(0)
    n_data_dims = x.size(-1)//2
    mask = x[:, :, n_data_dims:]
    utils.check_mask(x[:, :, :n_data_dims], mask)
    mask = (torch.sum(mask, -1, keepdim = True) > 0).float()
    assert(not torch.isnan(mask).any())
    return mask.squeeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None, z0_dim=None,
                 n_gru_units=100, device=torch.device('cpu')):
        super().__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        self.lstm = nn.LSTMCell(input_dim, latent_dim)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
           nn.Linear(latent_dim, 100),
           nn.Tanh(),
           nn.Linear(100, self.z0_dim * 2),)
        utils.init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True, save_info=False):
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        latent = self.run_odernn(data, time_steps, run_backwards)

        latent = latent.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:,-1] + 0.01, time_steps[:,-1]

        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)
            h = self.z0_diffeq_solver(h.unsqueeze(1), t).squeeze(1)
            h = repeat(h, "1 hid_dim -> BATCH_SIZE hid_dim", BATCH_SIZE=batch_size)
            

            xi = data[:,i,:]
            h_, c_ = self.lstm(xi, (h, c))
            mask = get_mask(xi)

            h = mask * h_ + (1 - mask) * h
            c = mask * c_ + (1 - mask) * c

            prev_t, t_i = time_steps[:,i], time_steps[:,i-1]

        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        decoder = nn.Sequential(nn.Linear(latent_dim, input_dim),)
        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)
