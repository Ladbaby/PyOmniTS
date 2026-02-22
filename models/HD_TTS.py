# Code from: https://github.com/Ladbaby/PyOmniTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *
from torch import Tensor

from layers.HD_TTS.lib.nn.models.hdtts_model import HDTTSModel
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    '''
    - paper: "Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling" (ICML 2024)
    - paper link: https://proceedings.mlr.press/v235/marisca24a.html
    - code adapted from: https://github.com/marshka/hdtts
    '''
    def __init__(
        self,
        configs: ExpConfigs
    ):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        self.hdtts = HDTTSModel(
            input_size=1,
            hidden_size=64,
            n_nodes=configs.enc_in,
            horizon=self.pred_len,
            rnn_layers=4,
            pooling_layers=1,
            exog_size=0,
            mask_size=1,
            dilation=3,
            cell="gru",
            mp_kernel_size=1,
            mp_method=["anisoconv", "propconv"],
            activation="elu"
        )

        # Initialize graph learner
        self.graph_learner = GraphLearner(configs.enc_in)

    def forward(
        self, 
        x: Tensor,
        x_mark: Tensor | None = None, 
        x_mask: Tensor | None = None, 
        y: Tensor | None = None, 
        y_mark: Tensor | None = None, 
        y_mask: Tensor | None = None,
        y_class: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        exp_stage: str = "train", 
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)
        
        u = None
        if (edge_index or edge_weight) is None:
            # We have to dynamically construct graph for input time series
            # Forward pass to get adjacency matrix
            adj_matrix = self.graph_learner().to(x.device)
            # Get edge index and weights
            edge_index, edge_weight = self.get_edge_index_and_weights(adj_matrix)
        # END adaptor

        '''
        - out: [batch_size, pred_len, c_out, 1]
        '''
        out, _, alpha, _ = self.hdtts(
            x = repeat(x, "b n f -> b n f 1"),
            edge_index = edge_index,
            edge_weight = edge_weight,
            input_mask = repeat(x_mask, "b n f -> b n f 1"),
            u = u
        )

        if self.configs.task_name in ["long_term_forecast", "short_term_forecast", "imputation"]:
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": out[:, -PRED_LEN:, f_dim:, 0],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError()


    def get_edge_index_and_weights(self, adj_matrix):
        n_variables = adj_matrix.size(0)
        # Create a meshgrid of indices
        row_idx, col_idx = torch.meshgrid(torch.arange(n_variables), torch.arange(n_variables), indexing='ij')
        edge_index = torch.stack([row_idx.flatten(), col_idx.flatten()], dim=0).to(adj_matrix.device)
        edge_weight = adj_matrix.flatten()
        return edge_index, edge_weight

class GraphLearner(nn.Module):
    def __init__(self, n_variables):
        super(GraphLearner, self).__init__()
        self.adjacency_matrix = nn.Parameter(torch.randn(n_variables, n_variables)).cuda()

    def forward(self):
        # Optionally apply some transformation
        adj_matrix = F.softmax(self.adjacency_matrix, dim=-1)  # Row-wise softmax
        return adj_matrix