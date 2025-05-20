from torch import nn
import torch

from layers.higp_lib.nn.utils import maybe_cat_emb
from layers.tsl.nn.blocks import RNN, MLPDecoder
from layers.tsl.nn.layers import DiffConv

from layers.tsl.nn.layers.base import NodeEmbedding
from layers.tsl.nn.models import BaseModel
from layers.tsl.nn.utils import maybe_cat_exog


class TimeThenSpaceModel(BaseModel):
    return_type = torch.Tensor

    r""""""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 emb_size: int,
                 temporal_layers: int,
                 ff_size: int = None,
                 rnn_size: int = None,
                 skip_connection: bool = False,
                 output_size: int = None,
                 exog_size: int = 0,
                 activation: str = 'elu'):
        super(TimeThenSpaceModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.horizon = horizon
        self.skip_connection = skip_connection

        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=emb_size)

        rnn_size = rnn_size or hidden_size

        self.input_encoder = nn.Linear(
            input_size + exog_size + emb_size,
            rnn_size,
        )

        if rnn_size != hidden_size:
            self.temporal_encoder = RNN(
                input_size=rnn_size,
                hidden_size=rnn_size,
                output_size=hidden_size,
                return_only_last_state=True,
                n_layers=temporal_layers
            )
        else:
            self.temporal_encoder = RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                return_only_last_state=True,
                n_layers=temporal_layers
            )

        decoder_input_size = hidden_size + emb_size
        if skip_connection:
            decoder_input_size += rnn_size

        ff_size = ff_size or hidden_size
        self.decoder = MLPDecoder(
            input_size=decoder_input_size,
            output_size=self.output_size,
            horizon=horizon,
            hidden_size=ff_size,
            activation=activation,
        )

    def message_passing(self, x, edge_index, edge_weight=None):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_weight=None, u=None):
        """"""
        x = maybe_cat_exog(x, u)

        emb = self.emb()
        x = maybe_cat_emb(x, emb)

        # temporal encoding
        # weights are shared across levels
        x = self.input_encoder(x)

        x = self.temporal_encoder(x)

        out = self.message_passing(x,
                                   edge_index=edge_index,
                                   edge_weight=edge_weight)

        out = maybe_cat_emb(out, emb)

        if self.skip_connection:
            out = torch.cat([out, x], dim=-1)

        out = self.decoder(out)
        return out
