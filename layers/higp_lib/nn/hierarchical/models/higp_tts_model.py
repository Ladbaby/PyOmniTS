from layers.higp_lib.nn.hierarchical.models.hierarchical_time_than_space import HierarchicalTimeThanSpaceModel
from layers.higp_lib.nn.hierarchical.pyramidal_gnn import PyramidalGNN

class HiGPTTSModel(HierarchicalTimeThanSpaceModel):
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
                 mode: str = 'gated',
                 skip_connection: bool = False,
                 top_down: bool = False,
                 output_size: int = None,
                 rnn_size: int = None,
                 ff_size: int = None,
                 exog_size: int = 0,
                 temporal_layers: int = 1,
                 gnn_layers: int = 1,
                 temp_decay: float = 0.99999,
                 activation: str = 'elu'):
        super(HiGPTTSModel, self).__init__(input_size=input_size,
                                           horizon=horizon,
                                           n_nodes=n_nodes,
                                           hidden_size=hidden_size,
                                           rnn_size=rnn_size,
                                           ff_size=ff_size,
                                           emb_size=emb_size,
                                           levels=levels,
                                           n_clusters=n_clusters,
                                           single_sample=single_sample,
                                           skip_connection=skip_connection,
                                           output_size=output_size,
                                           exog_size=exog_size,
                                           temporal_layers=temporal_layers,
                                           activation=activation,
                                           temp_decay=temp_decay)

        if top_down:
            assert skip_connection, "Top-down requires skip connection"

        self.gnn_layers = gnn_layers

        self.hier_gnn = PyramidalGNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            levels=levels,
            layers=gnn_layers,
            activation=activation,
            mode=mode
        )

    def hierarchical_message_passing(self, x, selects, edge_index, edge_weight, **kwargs):
        return self.hier_gnn(xs=x,
                             selects=selects,
                             edge_index=edge_index,
                             edge_weight=edge_weight)
