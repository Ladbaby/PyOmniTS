import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

from layers.higp_lib.nn.hierarchical.pyramidal_gnn import PyramidalGNN
from layers.higp_lib.nn.hierarchical.hierarchy_builders import MinCutHierarchyBuilder
from layers.higp_lib.nn.hierarchical.hierarchy_encoders import HierarchyEncoder
from layers.higp_lib.nn.hierarchical.ops import compute_aggregation_matrix
from layers.higp_lib.nn.utils import maybe_cat_emb
from layers.tsl.nn.blocks import RNN, MLPDecoder
from layers.tsl.nn.layers.base import NodeEmbedding
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class GraphLearner(nn.Module):
    def __init__(self, n_variables):
        super(GraphLearner, self).__init__()
        self.adjacency_matrix = nn.Parameter(torch.randn(n_variables, n_variables)).cuda()

    def forward(self):
        # Optionally apply some transformation
        adj_matrix = F.softmax(self.adjacency_matrix, dim=-1)  # Row-wise softmax
        return adj_matrix

class HierarchicalTimeThanSpaceModel(nn.Module):
    def __init__(self,
        configs: ExpConfigs,
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
        activation: str = 'silu'
    ):
        super(HierarchicalTimeThanSpaceModel, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.levels = levels
        self.horizon = horizon
        self.single_sample = single_sample
        self.skip_connection = skip_connection

        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=emb_size)
        rnn_size = rnn_size if rnn_size is not None else hidden_size

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

        enc_in_list = [configs.enc_in]
        for i in range(1, self.levels):
            enc_in_list.append(max(1, configs.enc_in // (10 ** i)))

        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.decoders = nn.ModuleList([MLPDecoder(
                input_size=decoder_input_size,
                output_size=self.output_size,
                horizon=horizon,
                hidden_size=ff_size,
                activation=activation,
            ) for _ in range(self.levels)])
        else:
            raise NotImplementedError

        # Initialize graph learner
        self.graph_learner = GraphLearner(configs.enc_in)

    def hierarchical_message_passing(self, x, **kwargs):
        raise NotImplementedError

    def get_edge_index_and_weights(self, adj_matrix):
        n_variables = adj_matrix.size(0)
        # Create a meshgrid of indices
        row_idx, col_idx = torch.meshgrid(torch.arange(n_variables), torch.arange(n_variables), indexing='ij')
        edge_index = torch.stack([row_idx.flatten(), col_idx.flatten()], dim=0).to(adj_matrix.device)
        edge_weight = adj_matrix.flatten()
        return edge_index, edge_weight

    def forward(
        self, 
        x: Tensor,
        y: Tensor = None,
        y_mask: Tensor = None, 
        y_class: Tensor = None, 
        edge_index: Tensor = None,
        edge_weight: Tensor = None,
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        u = None
        if None in [edge_index, edge_weight]:
            # We have to dynamically construct graph for input time series
            # Forward pass to get adjacency matrix
            adj_matrix = self.graph_learner().to(x.device)
            # Get edge index and weights
            edge_index, edge_weight = self.get_edge_index_and_weights(adj_matrix)
        else:
            edge_index = edge_index[0]
            edge_weight = edge_weight[0]
        # END adaptor

        x = repeat(x, "b n f -> b n f 1")

        # x.shape = [64, 12, 207, 1]
        # edge_index.shape = [2, 2626]
        # edge_weight.shape = [2626]
        # u.shape = [64, 12, 207 10]
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

        # for level=3, xs contains 3 tensors:
        # 0.shape = [64, 207, 32]
        # 1.shape = [64, 20, 32]
        # 2.shape = [64, 1, 32]

        outs = self.hierarchical_message_passing(x=xs,
                                                 adjs=adjs,
                                                 selects=selects,
                                                 edge_index=edge_index,
                                                 edge_weight=edge_weight,
                                                 aggregation_matrix=aggregation_matrix,
                                                 sizes=sizes)

        # for level=3, outs also contains 3 tensors:
        # 0.shape = [64, 207, 32]
        # 1.shape = [64, 20, 32]
        # 2.shape = [64, 1, 32]

        for i in range(self.levels):
            outs[i] = maybe_cat_emb(outs[i], embs[i])

        # for level=3, outs now become:
        # 0.shape = [64, 207, 64]
        # 1.shape = [64, 20, 64]
        # 2.shape = [64, 1, 64]

        if self.skip_connection:
            for i in range(self.levels):
                outs[i] = torch.cat([outs[i], xs[i]], dim=-1)

        # for level=3, outs now become:
        # 0.shape = [64, 12, 207, 1]
        # 1.shape = [64, 12, 20, 1]
        # 2.shape = [64, 12, 1, 1]

        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            # skip connection and decoder
            for i in range(self.levels):
                outs[i] = self.decoders[i](outs[i])

            out = torch.cat(outs[::-1], dim=-2)

            # out.shape = [64, 12, 228, 1]
            # aggregation_matrix.shape = [21, 207]
            # sizes.shape = [207, 20, 1]
            # reg_losses: Tuple = (0.0, 0.0)

            pred = out.squeeze(-1)[:, :, :self.configs.c_out]
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": pred[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError

class Model(HierarchicalTimeThanSpaceModel):
    '''
    - paper: "Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting" (ICML 2024)
    - paper link: https://proceedings.mlr.press/v235/cini24a.html
    - code adapted from: https://github.com/andreacini/higp
    '''
    def __init__(self, configs: ExpConfigs):
        input_size: int = 1
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        horizon: int = self.pred_len
        n_nodes: int = configs.enc_in
        hidden_size: int = 32
        emb_size: int = 32
        levels: int = 3
        n_clusters: int = max(1, configs.enc_in // 10) # 20 originally
        single_sample: bool = True
        mode: str = 'gated'
        skip_connection: bool = False
        top_down: bool = False
        output_size: int = 1
        rnn_size: int = None
        ff_size: int = None
        exog_size: int = 0 # originally 10, assuming related to the last dimension of u in forward
        temporal_layers: int = 1
        gnn_layers: int = 2
        temp_decay: float = 0.99995
        activation: str = 'elu'


        super(Model, self).__init__(
            configs=configs, 
            input_size=input_size,
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
            temp_decay=temp_decay
        )

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
