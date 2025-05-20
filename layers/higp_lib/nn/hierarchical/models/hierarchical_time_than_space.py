from einops import repeat
from torch import nn
import torch

from layers.higp_lib.nn.hierarchical.hierarchy_builders import MinCutHierarchyBuilder
from layers.higp_lib.nn.hierarchical.hierarchy_encoders import HierarchyEncoder
from layers.higp_lib.nn.hierarchical.ops import compute_aggregation_matrix
from layers.higp_lib.nn.utils import maybe_cat_emb
from layers.tsl.nn.blocks import RNN, MLPDecoder

from layers.tsl.nn.layers.base import NodeEmbedding
from layers.tsl.nn.models import BaseModel


class HierarchicalTimeThanSpaceModel(BaseModel):
    return_type = tuple

    r""""""
    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 emb_size: int,
                 levels: int,
                 n_clusters: int,
                 single_sample: bool,
                 skip_connection: bool = False,
                 output_size: int = None,
                 ff_size: int = None,
                 rnn_size: int = None,
                 exog_size: int = 0,
                 temporal_layers: int = 1,
                 temp_decay: float = 0.5,
                 activation: str = 'silu'):
        super(HierarchicalTimeThanSpaceModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.levels = levels
        self.horizon = horizon
        self.single_sample = single_sample
        self.skip_connection = skip_connection

        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=emb_size)
        rnn_size = rnn_size or hidden_size


        self.input_encoder = HierarchyEncoder(
            input_size=input_size,
            hidden_size=rnn_size,
            exog_size=exog_size,
            emb_size=emb_size
        )

        self.hierarchy_builder = MinCutHierarchyBuilder(
            n_nodes=n_nodes,
            hidden_size=emb_size,
            n_clusters=n_clusters,
            n_levels=levels,
            temp_decay=temp_decay
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
        self.decoders = nn.ModuleList([MLPDecoder(
            input_size=decoder_input_size,
            output_size=self.output_size,
            horizon=horizon,
            hidden_size=ff_size,
            activation=activation,
        ) for _ in range(self.levels)])

    def hierarchical_message_passing(self, x, **kwargs):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_weight=None, u=None):
        """"""
        emb = self.emb()
        if self.training and not self.single_sample:
            emb = repeat(emb, 'n f -> b n f', b=x.size(0))

        # extract hierarchy
        embs, \
        adjs, \
        selects, \
        sizes, \
        reg_losses = self.hierarchy_builder(emb,
                                            edge_index=edge_index,
                                            edge_weight=edge_weight)

        aggregation_matrix = compute_aggregation_matrix(selects)

        # temporal encoding
        # weights are shared across levels
        x = self.input_encoder(x=x,
                               u=u,
                               embs=embs,
                               selects=selects,
                               cat_output=True)

        x = self.temporal_encoder(x)
        xs = list(torch.split(x, sizes, dim=-2))

        outs = self.hierarchical_message_passing(x=xs,
                                                 adjs=adjs,
                                                 selects=selects,
                                                 edge_index=edge_index,
                                                 edge_weight=edge_weight,
                                                 aggregation_matrix=aggregation_matrix,
                                                 sizes=sizes)

        for i in range(self.levels):
            outs[i] = maybe_cat_emb(outs[i], embs[i])

        if self.skip_connection:
            for i in range(self.levels):
                outs[i] = torch.cat([outs[i], xs[i]], dim=-1)

        # skip connection and decoder
        for i in range(self.levels):
            outs[i] = self.decoders[i](outs[i])

        out = torch.cat(outs[::-1], dim=-2)

        return out, \
               aggregation_matrix, \
               sizes[::-1], \
               reg_losses