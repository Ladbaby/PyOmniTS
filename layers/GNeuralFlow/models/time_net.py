import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeIdentity(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, t):
        assert t.shape[-1] == 1
        return t.repeat_interleave(self.out_dim, dim=-1)

    def derivative(self, t):
        return torch.ones_like(t).repeat_interleave(self.out_dim, dim=-1)

class TimeLinear(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, out_dim))
        nn.init.xavier_uniform_(self.scale)

    def forward(self, t):
        return self.scale * t

    def derivative(self, t):
        return self.scale * torch.ones_like(t)


class TimeTanh(TimeLinear):
    def forward(self, t):
        return torch.tanh(self.scale * t)

    def derivative(self, t):
        return self.scale * (1 - self.forward(t)**2)

class TimeLog(TimeLinear):
    def forward(self, t):
        return torch.log(self.scale.exp() * t + 1)

    def derivative(self, t):
        return self.scale.exp() / (self.scale.exp() * t + 1)

class TimeFourier(nn.Module):
    """
    Fourier features: sum_k x_k sin(s_k t).

    Args:
        out_dim: Output dimension.
        hidden_dim: Number of fourier features.
        lmbd: Lambda parameter of exponential distribution used
            to initialize shift parameters. (default: 0.5)
    """
    def __init__(self, out_dim, hidden_dim, lmbd=0.5, bounded=False, **kwargs):
        super().__init__()
        self.bounded = bounded
        self.hidden_dim = hidden_dim
        self.shift = nn.Parameter(-torch.log(1 - torch.rand(out_dim, hidden_dim)) / lmbd)

        self.weight = nn.Parameter(torch.Tensor(out_dim, hidden_dim))
        nn.init.xavier_normal_(self.weight)

    def get_scale(self):
        if self.bounded:
            return F.softmax(self.weight, -1) / 2
        else:
            return self.weight / self.hidden_dim

    def forward(self, t):
        t = t.unsqueeze(-1)
        scale = self.get_scale()
        t = scale * torch.sin(self.shift * t)
        t = t.sum(-1)
        return t

    def derivative(self, t):
        t = t.unsqueeze(-1)
        scale = self.get_scale()
        t = self.shift * scale * torch.cos(self.shift * t)
        t = t.sum(-1)
        return t

class TimeFourierBounded(TimeFourier):
    """ Same as TimeFourier but between 0 and 1 """
    def __init__(self, out_dim, hidden_dim, lmbd=0.5, **kwargs):
        super().__init__(out_dim, hidden_dim, lmbd, True)
