from typing import Optional

from torch import Tensor, nn
from layers.tsl.nn import get_functional_activation
from layers.tsl.nn.layers import NodeEmbedding


class Encoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 mask_size: int = None,
                 n_nodes: int = None,
                 emb_size: int = None,
                 whiten_missing: bool = False,
                 activation: str = "linear"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.emb_size = emb_size or None
        self.exog_size = exog_size or None
        self.mask_size = mask_size or None

        if whiten_missing:
            assert mask_size == 1
        self.whiten_missing = whiten_missing

        self.lin_x = nn.Linear(input_size, hidden_size)

        if self.emb_size is not None:
            assert n_nodes is not None
            self.embeddings = NodeEmbedding(n_nodes, emb_size)
            self.lin_emb = nn.Linear(emb_size, hidden_size)
        else:
            self.register_parameter('embeddings', None)
            self.register_parameter('lin_emb', None)

        if self.mask_size is not None:
            self.lin_mask = nn.Linear(mask_size, hidden_size)
        else:
            self.register_parameter('lin_mask', None)

        if self.exog_size is not None:
            self.lin_u = nn.Linear(exog_size, hidden_size)
        else:
            self.register_parameter('lin_u', None)

        self.activation = get_functional_activation(activation)

    def compute_shape(self, source: Tensor, target: Tensor):
        src_size = list(source.size())
        for pos, size in enumerate(target.size()[:-1]):
            if src_size[pos] not in [1, size]:
                src_size.insert(pos, 1)
        return src_size

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                u: Optional[Tensor] = None):
        # x: [batch, *, node, *, features]
        out = self.lin_x(x)
        if self.embeddings is not None:
            emb = self.embeddings()
            emb = emb.view(self.compute_shape(emb, out))
            out = out + self.lin_emb(emb)
        if mask is not None:
            if mask.ndim != x.ndim:
                mask = mask.view(self.compute_shape(mask, out))
            if self.whiten_missing:
                out = out * mask  # apply zero-imputation
            out = out + self.lin_mask(mask.float())
        if u is not None:
            if u.ndim != x.ndim:
                u = u.view(self.compute_shape(u, out))
            out = out + self.lin_u(u)
        return self.activation(out)
