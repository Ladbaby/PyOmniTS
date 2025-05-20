import math
from typing import Optional, List

import torch
from torch import Tensor, nn
from layers.tsl.nn.blocks.encoders import RNNBase
from layers.tsl.nn.layers.recurrent import RNNCellBase, StateType


class DiagonalLinear(nn.Module):

    def __init__(self, in_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaimig uniform unrolled, for 1-dim vector
        bound = 1 / math.sqrt(self.in_features)
        gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
        w_bound = math.sqrt(3.0) * gain * bound
        nn.init.uniform_(self.weight, -w_bound, w_bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        output = input * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
        # w = torch.diag(self.weight)
        # return nn.functional.linear(input, w, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GRUDCellBase(RNNCellBase):
    """
    https://arxiv.org/abs/1606.01865

    x_mean: [n_nodes, input_size]
    forget_gate: [input_size+hidden_size+mask_size] x [hidden_size]
    update_gate: [input_size+hidden_size+mask_size] x [hidden_size]
    candidate_gate: [input_size+hidden_size+mask_size] x [hidden_size]
    decay_state: [mask_size] x [hidden_size]
    decay_input: [input_size] x [input_size]
    """

    def __init__(self, hidden_size: int,
                 forget_gate: nn.Module, update_gate: nn.Module,
                 candidate_gate: nn.Module, decay_state: nn.Module,
                 decay_input: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = forget_gate
        self.update_gate = update_gate
        self.candidate_gate = candidate_gate
        self.decay_state = decay_state
        self.decay_input = decay_input

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def reset_parameters(self):
        self.forget_gate.reset_parameters()
        self.update_gate.reset_parameters()
        self.candidate_gate.reset_parameters()
        self.decay_state.reset_parameters()
        if self.decay_input is not None:
            self.decay_input.reset_parameters()

    def initialize_state(self, x: Tensor) -> Tensor:
        return torch.zeros(x.size(0),
                           x.size(-2),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)

    def forward(self,
                x: Tensor,
                h: Tensor,
                x_last: Tensor,
                x_mean: Tensor,
                mask: Tensor,
                delta: Tensor,
                u: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        """
        Args:

            x: input with missing values [batch, *, input_size]
            h: previous state [batch, *, hidden_size]
            x_last: last observed value [batch, *, input_size]
            x_mean: average observed value [batch, *, input_size]
            mask: mask [batch, *, input_size]
            delta: interval from last observation [batch, *, input_size]
            **kwargs: gates forward keyword arguments
        """

        # Input decay
        if self.decay_input is not None:
            gamma_x = torch.exp(-torch.relu(self.decay_input(delta)))
            x_c = (gamma_x * x_last) + (1 - gamma_x) * x_mean
            x = torch.where(mask, x, x_c)

        # State decay
        gamma_h = torch.exp(-torch.relu(self.decay_state(delta)))
        h = gamma_h * h

        # Prepare inputs
        inputs = [h, x, mask]
        if u is not None:
            if u.ndim != x.ndim:
                u = u.unsqueeze(-2).expand(-1, x.size(-2), -1)
            inputs.append(u)
        x_gates = torch.cat(inputs, dim=-1)

        # Standard GRU
        r = torch.sigmoid(self.forget_gate(x_gates, **kwargs))
        u = torch.sigmoid(self.update_gate(x_gates, **kwargs))
        x_c = torch.cat([r * h] + inputs[1:], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c, **kwargs))
        h_new = u * h + (1. - u) * c

        return h_new


class GRUD(RNNBase):
    """"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 n_layers: int = 1,
                 x_mean: Optional[Tensor] = None,
                 *args, **kwargs):

        mask_size = input_size
        cells = []

        for i in range(n_layers):
            gate_in_size = input_size + hidden_size + mask_size + exog_size
            forget_gate = nn.Linear(gate_in_size, hidden_size)
            update_gate = nn.Linear(gate_in_size, hidden_size)
            candidate_gate = nn.Linear(gate_in_size, hidden_size)
            decay_state = nn.Linear(mask_size, hidden_size)
            decay_input = DiagonalLinear(input_size) if i == 0 else None

            cells.append(GRUDCellBase(hidden_size=hidden_size,
                                      forget_gate=forget_gate,
                                      update_gate=update_gate,
                                      candidate_gate=candidate_gate,
                                      decay_state=decay_state,
                                      decay_input=decay_input))
            input_size = hidden_size

        super().__init__(cells, *args, **kwargs)

        if x_mean is not None and x_mean.ndim == 2:
            x_mean = x_mean.unsqueeze(0)  # unsqueeze batch dim
        self.register_buffer('x_mean', x_mean)

    def get_delta(self, mask, sequence=None):
        """
        mask shape: [time_steps, *]
        sequence shape: [time_steps]

        Example:
            mask: [[1,1,0,1,0,1,1],
                   [0,1,1,0,0,0,1]]
            sequence: [0,0.1,0.6,1.6,2.2,2.5,3.1]
            delta: [[0.0, 0.1, 0.5, 1.5, 0.6, 0.9, 0.6],
                    [0.0, 0.1, 0.5, 1.0, 1.6, 1.9, 2.5]]
        """
        mask = mask.float()

        delta = torch.zeros_like(mask)
        time_steps = mask.size(1)

        if sequence is None:
            sequence = torch.arange(mask.size(1),
                                    dtype=mask.dtype, device=mask.device)

        for i in range(1, time_steps):
            delta[:, i] = (sequence[i] - sequence[i - 1] +
                           (1 - mask[:, i - 1]) * delta[:, i - 1])

        return delta

    def forward(self,  # noqa
                x: Tensor,
                mask: Tensor,
                x_mean: Tensor = None,
                u: Tensor = None,
                h: Optional[List[StateType]] = None):

        # Compute time intervals
        d = self.get_delta(mask)

        # Initialize state
        if h is None:
            h = self.initialize_state(x)

        # Get x_mean
        if x_mean is None:
            if self.x_mean is not None:  # get cached mean, if any
                x_mean = self.x_mean
            else:  # compute empirical mean of input sequence over time
                mask_sum = mask.sum(1)
                # b t n f -> b n f
                x_mean = torch.where(mask_sum > 0, (x * mask).sum(1) / mask_sum,
                                     torch.zeros_like(mask_sum))

        # Get x_last
        x_last = x_mean

        out = []
        steps = x.size(1)
        for step in range(steps):
            # save state before seeing input at time t
            if self.cat_states_layers:
                h_out = torch.cat(h, dim=-1)
            else:  # or take last layer's state
                h_out = h[-1]
            # save states for imputations as 1-step-ahead predictions
            out.append(h_out)
            u_s = u[:, step] if u is not None else None
            # update state
            h = self.single_pass(x[:, step], h, mask=mask[:, step],
                                 x_last=x_last, x_mean=x_mean, u=u_s,
                                 delta=d[:, step])
            # update last observed value
            x_last = torch.where(mask[:, step], x[:, step], x_last)

        if self.return_only_last_state:
            # out: [batch, *, features]
            return torch.cat(h, dim=-1) if self.cat_states_layers else h[-1]
        # out: [batch, time, *, features]
        out = torch.stack(out, dim=1)
        return out, h
