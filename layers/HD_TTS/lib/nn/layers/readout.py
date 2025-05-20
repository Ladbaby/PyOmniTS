from typing import Optional

from einops.layers.torch import Rearrange
from torch import nn, Tensor
from layers.tsl.nn.layers import Dense, MultiLinear


class Readout(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int = None,
                 horizon: int = None,
                 multi_readout: bool = False,
                 n_hidden_layers: int = 0,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon

        readout_input_size = input_size if n_hidden_layers == 0 else hidden_size

        # Add layers in reverse order, starting from the readout
        layers = []

        # Last linear layer
        if not multi_readout:
            # after transformation, reshape to have "... time nodes features"
            if horizon is not None:
                layers.append(Rearrange('... n (h f) -> ... h n f', h=horizon))
            else:
                horizon = 1
            layers.append(nn.Linear(readout_input_size, output_size * horizon))
        else:
            assert horizon is not None
            layers.append(MultiLinear(readout_input_size, output_size,
                                      n_instances=horizon,
                                      instance_dim=-3))

        # Optionally add hidden layers
        for i in range(n_hidden_layers):
            layers.append(
                Dense(input_size if i == (n_hidden_layers - 1) else hidden_size,
                      output_size=hidden_size,
                      activation=activation,
                      dropout=dropout)
            )

        self.mlp = nn.Sequential(*reversed(layers))

    def forward(self, x: Tensor):
        # x: [*, nodes, features] or x: [*, horizon, nodes, features]
        x = self.mlp(x)
        return x


class AttentionReadout(nn.Module):

    def __init__(self,
                 input_size: int,
                 dim_size: int,  # num elements on which to apply attention
                 horizon: int = None,
                 dim: int = -2,  # dimension along which to apply attention
                 output_size: int = None,
                 hidden_size: int = None,
                 mask_size: int = None,
                 fully_connected: bool = False,
                 multi_step_scores: bool = True,
                 ff_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.dim_size = dim_size
        self.horizon = horizon
        self.fully_connected = fully_connected
        self.multi_step_scores = multi_step_scores

        horizon = horizon or 1
        out_f = horizon if multi_step_scores else 1

        if fully_connected:
            # b n l f -> b n (l h) f
            self.lin_scores_state = MultiLinear(dim_size, dim_size * out_f,
                                                n_instances=input_size,
                                                instance_dim=-1,
                                                channel_dim=-2)
        else:
            # b n l f -> b n l h
            self.lin_scores_state = nn.Linear(input_size, out_f)

        if mask_size is not None:
            # mask: [batch nodes features]
            if fully_connected:
                self.lin_scores_mask = nn.Sequential(
                    nn.Linear(mask_size, dim_size * out_f * input_size),
                    Rearrange('b n (l h f) -> b n (l h) f',
                              l=dim_size, f=out_f),
                )
            else:
                self.lin_scores_mask = nn.Sequential(
                    nn.Linear(mask_size, dim_size * out_f),
                    Rearrange('b n (l h) -> b n l h', l=dim_size, f=out_f),
                )
        else:
            self.register_parameter('lin_scores_mask', None)

        # Rearrange scores to have the same shape as the input
        self.rearrange = nn.Identity()
        if multi_step_scores and fully_connected:
            self.rearrange = Rearrange('b n (l h) f -> b h n l f', l=dim_size)
        elif multi_step_scores:
            self.rearrange = Rearrange('b n l h -> b h n l 1', l=dim_size)

        if output_size is not None:
            self.readout = Readout(input_size=input_size,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   horizon=self.horizon,
                                   multi_readout=self.multi_step_scores,
                                   n_hidden_layers=ff_layers - 1,
                                   activation=activation,
                                   dropout=dropout)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        # mask: [batch, nodes, features]
        if self.dim != -2:
            x = x.movedim(self.dim, -2)
        # x: [batch, nodes, layers, features]

        # Compute scores from features with a linear reduction
        scores = self.lin_scores_state(x)  # -> [batch, nodes, layers, out_f]

        # Optionally add mask information inside the score
        if self.lin_scores_mask is not None:
            scores = scores + self.lin_scores_mask(mask)

        # Normalize scores with softmax
        scores = self.rearrange(scores)
        alpha = scores.softmax(-2)  # -> [batch, *, nodes, layers, features]

        # Aggregate along layers dimension (self.dim) according to the scores
        if self.multi_step_scores:
            x = x.unsqueeze(-4)  # apply different score at each (layer, step)
        x = (x * alpha).sum(-2)  # -> [batch, *, nodes, features]

        if self.dim != -2:
            alpha = alpha.movedim(-2, self.dim)
        alpha = alpha.mean(-1)  # ... l 1 -> ... l ...

        if self.readout is None:
            return x, alpha

        return self.readout(x), x, alpha
