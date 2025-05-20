"""GNN taken from GANF https://github.com/EnyanDai/GANF"""
import torch.nn as nn
import torch



class GNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        # self.lin_n = nn.Linear(input_size, hidden_size)
        # self.lin_r = nn.Linear(hidden_size, input_size, bias=False)
        self.lin_n = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   # nn.Linear(hidden_size, hidden_size),
                                   # nn.Linear(hidden_size, hidden_size * 2),
                                   # nn.Linear(hidden_size * 2, hidden_size),
                                   nn.Linear(hidden_size, hidden_size))

        self.lin_r = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                   # nn.Linear(hidden_size, hidden_size),
                                   # nn.Linear(hidden_size, hidden_size*2),
                                   # nn.Linear(hidden_size*2, hidden_size),
                                   nn.Linear(hidden_size, input_size))

        self.act = nn.ReLU()

    def forward(self, h, a):
        """
        Args:
            h: data: N,L,K,D - N:batchsize, L: num of timesteps, K: number of sensors, D: feature dimension
            a: Adjacency matrix K,K
        Returns: vector of dimension N,L,K,D
        """
        h_n = self.lin_n(torch.einsum('nlkd,kj->nljd', h, a))
        h = self.lin_r(self.act(h_n))
        return h
