import torch
import torch_sparse
from torch import nn, Tensor
from torch_sparse import SparseTensor


class PropConv(nn.Module):
    """Propagates the input signal through the graph, without any
    transformation of the input signal.

    Args:
        in_channels (int, optional): The number of input channels.
            (default: ``None``)
        out_channels (int, optional): The number of output channels.
            (default: ``None``)
        add_backward (bool, optional): If ``True``, then features are
            propagated in both directions of the edges, by splitting
            the input tensor in half and propagating each half in
            opposite directions.
            (default: ``True``)
        use_edge_weights (bool, optional): If ``True``, then the edge weights
            are used in the propagation.
            (default: ``True``)
        normalize_weights (bool, optional): If ``True``, then the edge weights
            are normalized before being used in each propagation.
            (default: ``True``)
    """

    def __init__(self, in_channels: int = None,  # added for compatibility
                 out_channels: int = None,       # with the other layers
                 add_backward: bool = True,
                 use_edge_weights: bool = True,
                 normalize_weights: bool = True):
        super().__init__()
        self.in_channels = (in_channels // 2) if add_backward else in_channels
        self.out_channels = out_channels or in_channels
        self.add_backward = add_backward
        self.use_edge_weights = use_edge_weights
        self.normalize_weights = normalize_weights
        self.reduce = 'mean' if self.normalize_weights else 'sum'

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if not self.use_edge_weights:
            adj_t = adj_t.set_value(None)
        if self.add_backward:
            x_fwd = torch_sparse.matmul(adj_t, x[..., :self.in_channels],
                                        reduce=self.reduce)
            x_bwd = torch_sparse.matmul(adj_t.t(), x[..., self.in_channels:],
                                        reduce=self.reduce)
            return torch.cat([x_fwd, x_bwd], dim=-1)
        return torch_sparse.matmul(adj_t, x, reduce=self.reduce)
