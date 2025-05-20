from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from layers.tsl.nn import maybe_cat_exog


class DRNN(nn.Module):
    """The Dilated Recurrent Neural Network encoder from the paper `"Dilated
    Recurrent Neural Networks" <https://arxiv.org/abs/1710.02224>`_ (Chang et
    al., NeurIPS 2017).

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        dilation (int): Dilation factor.
        n_layers (int): Number of hidden layers.
        exog_size (int, optional): Size of the optional exogenous variables.
            (default: ``None``)
        output_size (int, optional): Size of the optional readout.
            (default: ``None``)
        return_only_last_state (bool, optional): Whether to return only the last
            state of the sequence. (default: ``False``)
        cell (str): Type of cell that should be use (options:``'gru'``,
            ``'lstm'``). (default: ``'gru'``)
        bias (bool): If ``False``, then the layer does not use bias
            weights. (default: ``True``)
        dropout (float): Dropout probability. (default: ``0.``)
    """

    def __init__(self, input_size: int, hidden_size: int, dilation: int,
                 n_layers: int,
                 exog_size: int = None,
                 output_size: int = None,
                 return_only_last_state: bool = False,
                 cell: str = 'gru',
                 bias: bool = True,
                 dropout: float = 0.):
        super(DRNN, self).__init__()
        self.return_only_last_state = return_only_last_state
        self.dilation = dilation
        self.receptive_field = dilation ** n_layers + 1

        if cell == 'gru':
            cell = nn.GRU
        elif cell == 'lstm':
            cell = nn.LSTM
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        if exog_size is not None:
            input_size += exog_size

        self.rnns = nn.ModuleList([
            cell(input_size=input_size if i == 0 else hidden_size,
                 hidden_size=hidden_size,
                 num_layers=1,
                 bias=bias,
                 dropout=dropout)
            for i in range(n_layers)
        ])

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor, u: Optional[Tensor] = None):
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        # x: [batches, steps, nodes, features]
        x = maybe_cat_exog(x, u)
        b, num_steps, *_ = x.size()
        x = rearrange(x, 'b s n f -> s (b n) f')

        out = []
        for rnn in self.rnns:
            x, *_ = rnn(x)
            if self.return_only_last_state:
                out.append(x[-1])
            else:
                raise NotImplementedError
            # Downsample the input sequence
            num_steps = x.size(0)
            # Ensure last element is the last also in the downsampled sequence
            offset = (num_steps - 1) % self.dilation
            x = x[offset::self.dilation]

        out = torch.stack(out, dim=0)
        out = rearrange(out, 'l ... (b n) f -> b l ... n f', b=b)
        if self.readout is not None:
            return self.readout(out)
        return out
