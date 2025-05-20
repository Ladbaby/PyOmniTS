import torch
import torch.nn as nn
import torch.nn.functional as F

import layers.GNeuralFlow.experiments.latent_ode.lib.utils as utils
from layers.GNeuralFlow.models.gnn import GNN
from layers.GNeuralFlow.models.mlp import MLP


def get_mask(x):
    x = x.unsqueeze(0)
    n_data_dims = x.size(-1) // 2
    mask = x[:, :, n_data_dims:]
    utils.check_mask(x[:, :, :n_data_dims], mask)
    mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
    assert (not torch.isnan(mask).any())
    return mask.squeeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_dim,
        z0_diffeq_solver=None,
        z0_dim=None,
        n_gru_units=100,
        device=torch.device('cpu'),
        nfeats=None,
        nsens=None,
        dim=None,
        enc_type=None,
    ):
        super().__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        self.nfeats = nfeats
        self.n_sensors = nsens
        self.dim = dim
        self.enc = enc_type

        self.lstm = nn.LSTMCell(input_dim, latent_dim)
        self.lstm2 = nn.LSTMCell(input_dim, latent_dim)

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2), )
        utils.init_network_weights(self.transform_z0)

        self.gcn = nn.ModuleList([GNN(input_size=1, hidden_size=32)]) # WARNING: this GNN is somewhat wired, since the input_size is n_feats during initialization, but the actual input match the input_size with latent dimension

    def forward(self, truth, mask, time_steps, adj=None, run_backwards=True, save_info=False):
        assert (not torch.isnan(truth).any())
        assert (not torch.isnan(mask).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, _, _ = truth.size()

        if self.enc == 'rnn2':
            data = torch.cat((truth, mask), -1)
            latent = self.run_odernn2(data, time_steps, adj, run_backwards)
        else:
            latent = self.run_odernn(
                truth, mask, time_steps, adj, run_backwards)

        latent = latent.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = self.transform_z0(latent).chunk(2, dim=-1)
        std_z0 = F.softplus(std_z0)

        return mean_z0, std_z0

    def run_odernn(self, truth, mask, time_steps, adj, run_backwards=True):
        data = torch.cat((truth, mask), -1)

        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]
        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        x = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)
        c2 = torch.zeros(batch_size, self.latent_dim).to(data)
        h = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)
            x = self.z0_diffeq_solver(x=x.unsqueeze(1),
                                      h=h.unsqueeze(1),
                                      t=t).squeeze(1)

            xi = data[:, i, :]
            if adj is not None:
                xi_truth = truth[:, i, :]
                x_gcn = xi_truth.view(
                    batch_size, 1, self.n_sensors, self.nfeats)
                _h = self.gcn(h=x_gcn, a=adj)
                _h = _h.view(batch_size, self.nfeats * self.n_sensors)
                h_true = torch.mul(_h, xi[:, self.dim:])
                h_mask = xi[:, self.dim:]
                hi = torch.cat((h_true, h_mask), -1)

            h_, c_ = self.lstm(xi, (x, c))
            mask = get_mask(xi)
            x = mask * h_ + (1 - mask) * x
            c = mask * c_ + (1 - mask) * c

            if adj is not None:
                h2_, c2_ = self.lstm2(hi, (x, c2))
                mask = get_mask(xi)
                h = mask * h2_ + (1 - mask) * x
                c2 = mask * c2_ + (1 - mask) * c2

            prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

        return x

    def run_odernn2(self, data, time_steps, adj, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)
            h = self.z0_diffeq_solver(x=h.unsqueeze(1), t=t).squeeze(1)

            xi = data[:, i, :]
            if adj is None:
                h_, c_ = self.lstm(xi, (h, c))
                mask = get_mask(xi)
            else:
                x_dim = xi.shape[-1] // 2
                a_dim = adj.shape[0]
                x_truth = xi[:, :x_dim]
                x_mask = xi[:, x_dim:]
                x_gcn = x_truth if x_dim == a_dim else x_truth.view(
                    -1, a_dim, x_dim // a_dim)
                x_gcn = x_gcn if len(x_gcn.shape) == 3 else x_gcn.unsqueeze(2)
                emb = x_gcn.unsqueeze(1)
                for gnn in self.gcn:
                    emb = gnn(h=emb, a=adj)
                emb = emb.squeeze().view(-1, x_dim)
                xi_copy = xi.clone()
                xi_copy[:, :x_dim] = x_mask * emb
                h_, c_ = self.lstm(xi_copy, (h, c))
                mask = get_mask(xi_copy)

            h = mask * h_ + (1 - mask) * h
            c = mask * c_ + (1 - mask) * c

            prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

        return h

    def run_odernn3(self, data, time_steps, adj, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)

            _h = h.clone()
            for ix in range(len(self.gcn)):
                _h = self.latent2nsens[ix](_h)
                _h = self.gcn[ix](_h.unsqueeze(1).unsqueeze(-1), adj).squeeze()
                _h = self.nsens2latent[ix](_h)

            h = self.z0_diffeq_solver(x=h.unsqueeze(
                1), adj=_h.unsqueeze(1), t=t).squeeze(1)

            xi = data[:, i, :]
            h_, c_ = self.lstm(xi, (h, c))
            mask = get_mask(xi)

            h = mask * h_ + (1 - mask) * h
            c = mask * c_ + (1 - mask) * c

            prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

        return h

    def run_odernn_orig(self, data, time_steps, run_backwards=True):
        batch_size, n_tp, n_dims = data.size()
        prev_t, t_i = time_steps[:, -1] + 0.01, time_steps[:, -1]

        time_points_iter = range(0, time_steps.shape[1])
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        h = torch.zeros(batch_size, self.latent_dim).to(data)
        c = torch.zeros(batch_size, self.latent_dim).to(data)

        for i in time_points_iter:
            t = (t_i - prev_t).unsqueeze(1)
            h = self.z0_diffeq_solver(h.unsqueeze(1), t).squeeze(1)

            xi = data[:, i, :]
            h_, c_ = self.lstm(xi, (h, c))
            mask = get_mask(xi)

            h = mask * h_ + (1 - mask) * h
            c = mask * c_ + (1 - mask) * c

            prev_t, t_i = time_steps[:, i], time_steps[:, i - 1]

        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), )
        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)
