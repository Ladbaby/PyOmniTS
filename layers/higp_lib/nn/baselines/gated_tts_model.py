from torch import nn

from layers.higp_lib.nn.layers import GraphAnisoConv

from layers.higp_lib.nn.base.time_then_space_model import TimeThenSpaceModel

class GatedTTSModel(TimeThenSpaceModel):
    r""""""
    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 gnn_layers: int,
                 emb_size: int,
                 temporal_layers: int,
                 output_size: int = None,
                 exog_size: int = 0,
                 activation: str = 'elu',
                 skip_connection: bool = False
                 ):
        super(GatedTTSModel, self).__init__(input_size=input_size,
                                            horizon=horizon,
                                            n_nodes=n_nodes,
                                            hidden_size=hidden_size,
                                            emb_size=emb_size,
                                            temporal_layers=temporal_layers,
                                            output_size=output_size,
                                            exog_size=exog_size,
                                            activation=activation,
                                            skip_connection=skip_connection
                                            )

        self.gnn_layers = nn.ModuleList([
            GraphAnisoConv(
                input_size=hidden_size,
                output_size=hidden_size,
                activation=activation) for _ in range(gnn_layers)
        ])

    def message_passing(self, x, edge_index, edge_weight=None):
        for layer in self.gnn_layers:
            x = layer(x, edge_index=edge_index)
        return x