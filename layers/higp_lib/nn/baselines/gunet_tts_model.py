from torch import nn

from layers.higp_lib.nn.base.time_then_space_model import TimeThenSpaceModel
from layers.higp_lib.nn.layers.graph_unet import GraphUNet


class GUNetTTSModel(TimeThenSpaceModel):
    r""""""

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 pooling_factor: float,
                 gnn_layers: int,
                 emb_size: int,
                 temporal_layers: int,
                 ff_size: int = None,
                 rnn_size: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 activation: str = 'elu',
                 skip_connection: bool = False):
        super(GUNetTTSModel, self).__init__(input_size=input_size,
                                            horizon=horizon,
                                            n_nodes=n_nodes,
                                            rnn_size=rnn_size,
                                            hidden_size=hidden_size,
                                            ff_size=ff_size,
                                            emb_size=emb_size,
                                            temporal_layers=temporal_layers,
                                            output_size=output_size,
                                            exog_size=exog_size,
                                            activation=activation,
                                            skip_connection=skip_connection
                                            )

        self.gnn = GraphUNet(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=gnn_layers,
            pooling_factor=pooling_factor,
            activation=activation
        )

    def message_passing(self, x, edge_index, edge_weight=None):
        return self.gnn(x=x, edge_index=edge_index, edge_weight=edge_weight)
