import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(nn.Module):
    '''
    - paper: "HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting" (ICML 2025)
    - paper link: https://openreview.net/forum?id=u8wRbX2r2V
    - code adapted from: https://github.com/Ladbaby/PyOmniTS

    Note: HyperIMTS expects the unpadded input the same as SeFT and GraFITi, so we refer to some of the codes from these models when converting padded samples to unpadded ones and doing attention on flattened tensors. 
    '''
    def __init__(
        self,
        configs: ExpConfigs
    ):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = configs.n_layers
        self.n_heads = configs.n_heads

        seq_len = configs.seq_len_max_irr or configs.seq_len
        pred_len = configs.pred_len_max_irr or configs.pred_len
        self.hypergraph_encoder = HypergraphEncoder(
            enc_in=self.enc_in,
            time_length=seq_len + pred_len,
            d_model=self.d_model
        )
        self.hypergraph_learner = HypergraphLearner(
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            time_length=seq_len + pred_len,
        )

        self.hypergraph_decoder = nn.Linear(3*self.d_model, 1)

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None,
        x_L_flattened: Tensor = None,
        x_y_mask_flattened: Tensor = None,
        y_L_flattened: Tensor = None,
        y_mask_L_flattened: Tensor = None,
        exp_stage: str = "train", 
        **kwargs
    ):
        # adaptor for unified pipeline input shape
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.configs.pred_len if self.configs.pred_len != 0 else SEQ_LEN
        if x_mark is None:
            x_mark = repeat(torch.arange(end=x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "L -> B L 1", B=x.shape[0])
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        _, PRED_LEN, _ = y.shape
        L = SEQ_LEN + PRED_LEN

        x_zeros = torch.zeros_like(y, dtype=x.dtype, device=x.device) # fill unknown forecast targets with zeros for input
        y_zeros = torch.zeros_like(x, dtype=y.dtype, device=y.device)
        x_y_mark = torch.cat([x_mark, y_mark], dim=1) # (BATCH_SIZE, L, 1)
        x_L = torch.cat([x, x_zeros], dim=1)
        x_y_mask = torch.cat([x_mask, y_mask], dim=1)
        y_L = torch.cat([y_zeros, y], dim=1)
        y_mask_L = torch.cat([y_zeros, y_mask], dim=1)
        # END adaptor

        time_indices = torch.cumsum(torch.ones_like(x_L).to(torch.int64), dim=1) - 1  # (BATCH_SIZE, L, ENC_IN) init for time indices. 0, 1, 2...
        variable_indices = torch.cumsum(torch.ones_like(x_L).to(torch.int64), dim=-1) - 1  # (BATCH_SIZE, L, ENC_IN) init for variable indices. 0, 1, 2...
        x_y_mask_bool = x_y_mask.to(torch.bool)  # (BATCH_SIZE, L, ENC_IN)

        # adaptor for extensibility, if the input is padded (e.g., MTS), we flatten it
        # BEGIN adaptor

        # to enable batch learning for multiple samples, not meant for alignment within sample
        def pad(v): return F.pad(v, [0, N_OBSERVATIONS_MAX - len(v)], value=0)
        if (x_L_flattened or x_y_mask_flattened or y_L_flattened or y_mask_L_flattened) is None:
            # get total number of observations in each sample, and take max
            N_OBSERVATIONS_MAX = torch.max(x_y_mask.sum((1, 2))).to(torch.int64)

            # flatten everything, from (L, ENC_IN) to (N_OBSERVATIONS_MAX), where observations belonging to the same timestep are nearby
            # note that r[m] won't keep the original tensor shape by default, thus flattened
            x_L_flattened = torch.stack([pad(r[m]) for r, m in zip(x_L, x_y_mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
            x_y_mask_flattened = torch.stack([pad(r[m]) for r, m in zip(x_y_mask, x_y_mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)

            y_L_flattened = torch.stack([pad(r[m]) for r, m in zip(y_L, x_y_mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
            y_mask_L_flattened = torch.stack([pad(r[m]) for r, m in zip(y_mask_L, x_y_mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        # END adaptor

        time_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(time_indices, x_y_mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        variable_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(variable_indices, x_y_mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)

        # IMTS to hypergraph
        (
            observation_nodes, 
            temporal_hyperedges, 
            variable_hyperedges, 
            temporal_incidence_matrix, 
            variable_incidence_matrix
        ) = self.hypergraph_encoder(
            x_L_flattened=x_L_flattened,
            x_y_mask_flattened=x_y_mask_flattened,
            y_mask_L_flattened=y_mask_L_flattened,
            x_y_mark=x_y_mark,
            variable_indices_flattened=variable_indices_flattened,
            time_indices_flattened=time_indices_flattened,
            N_OBSERVATIONS_MAX=N_OBSERVATIONS_MAX
        )

        # hypergraph learning
        (
            observation_nodes, 
            temporal_hyperedges, 
            variable_hyperedges
        ) = self.hypergraph_learner(
            observation_nodes=observation_nodes, 
            temporal_hyperedges=temporal_hyperedges, 
            variable_hyperedges=variable_hyperedges,
            time_indices_flattened=time_indices_flattened,
            variable_indices_flattened=variable_indices_flattened,
            temporal_incidence_matrix=temporal_incidence_matrix, 
            variable_incidence_matrix=variable_incidence_matrix,
            x_y_mask_flattened=x_y_mask_flattened,
            x_y_mask=x_y_mask,
            y_mask_L_flattened=y_mask_L_flattened
        )

        # hypergraph to IMTS
        pred_flattened = self.hypergraph_decoder(
            torch.cat([
                observation_nodes, 
                temporal_hyperedges.gather(dim=1, index=repeat(
                    time_indices_flattened,
                    "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                    D=self.d_model
                )), 
                variable_hyperedges.gather(dim=1, index=repeat(
                    variable_indices_flattened,
                    "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                    D=self.d_model
                )),
            ], dim=-1)
        ).squeeze(-1)

        if exp_stage in ["train", "val"]:
            return {
                "pred": pred_flattened,
                "true": y_L_flattened,
                "mask": y_mask_L_flattened
            }
        else:
            # convert unpadded tensor back to shape [batch_size, seq_len + pred_len, enc_in] to align with the pipeline's unified api when testing
            pred = self.unpad_and_reshape(
                tensor_flattened=pred_flattened,
                original_mask=torch.cat([x_mask, y_mask], dim=1),
                original_shape=(BATCH_SIZE, SEQ_LEN + PRED_LEN, ENC_IN)
            )
            f_dim = -1 if self.configs.features == 'MS' else 0
            return {
                "pred": pred[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }

    def unpad_and_reshape(
        self, 
        tensor_flattened: Tensor, 
        original_mask: Tensor, 
        original_shape: Tuple
    ):
        batch_size, _, _ = original_shape
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype, device=tensor_flattened.device)

        for i in range(batch_size):
            masked_indices = original_mask[i].view(-1).nonzero(as_tuple=True)[0]
            unpadded_sequence = tensor_flattened[i][:len(masked_indices)]
            result[i].view(-1)[masked_indices] = unpadded_sequence
            
        return result
    
class HypergraphEncoder(nn.Module):
    '''
    IMTS to Hypergraph nodes and hyperedges
    - observed values -> nodes
    - timestamps -> temporal hyperedges
    - variables -> variable hyperedges
    '''
    def __init__(
        self,
        enc_in,
        time_length,
        d_model
    ):
        super().__init__()
        self.enc_in = enc_in
        self.time_length = time_length
        self.d_model = d_model

        self.variable_hyperedge_weights = nn.Parameter(
            torch.randn(enc_in, d_model),
            requires_grad=True
        )

        self.relu = nn.ReLU()
        self.observation_node_encoder = nn.Linear(2, d_model)
        self.temporal_hyperedge_encoder = nn.Linear(1, d_model)

    def forward(
        self,
        x_L_flattened: Tensor,
        x_y_mask_flattened: Tensor,
        y_mask_L_flattened: Tensor,
        x_y_mark: Tensor,
        variable_indices_flattened: Tensor,
        time_indices_flattened: Tensor,
        N_OBSERVATIONS_MAX: int
    ):
        BATCH_SIZE = x_L_flattened.shape[0]
        ENC_IN = self.enc_in
        L = x_y_mark.shape[1]

        # add indicator for forecast targets
        x_L_flattened = torch.stack(
            [
                x_L_flattened, 
                1 - x_y_mask_flattened + y_mask_L_flattened
            ],
            dim=-1
        ) # (BATCH_SIZE, N_OBSERVATIONS_MAX) -> (BATCH_SIZE, N_OBSERVATIONS_MAX, 2)

        # (BATCH_SIZE, L, N_OBSERVATIONS_MAX)
        # indicate for every temporal hyperedge, which observation connected to it
        temporal_incidence_matrix = repeat(
            time_indices_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE L N_OBSERVATIONS_MAX",
            L=L
        )
        temporal_incidence_matrix = (temporal_incidence_matrix == repeat(
            torch.ones(BATCH_SIZE, L).to(x_L_flattened.device).cumsum(dim=1),
            "BATCH_SIZE L -> BATCH_SIZE L N_OBSERVATIONS_MAX",
            N_OBSERVATIONS_MAX=N_OBSERVATIONS_MAX
        ) - 1).to(torch.float32)
        temporal_incidence_matrix = temporal_incidence_matrix * repeat(
            x_y_mask_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE L N_OBSERVATIONS_MAX", 
            L=L
        ) # remove non-observed values at tail

        
        # (BATCH_SIZE, ENC_IN, N_OBSERVATIONS_MAX)
        # indicate for every variable hyperedge, which observation connected to it
        variable_incidence_matrix = repeat(
            torch.ones([BATCH_SIZE, ENC_IN]).cumsum(dim=1).to(x_L_flattened.device) - 1,
            "BATCH_SIZE ENC_IN -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            N_OBSERVATIONS_MAX=N_OBSERVATIONS_MAX
        )
        variable_incidence_matrix = (variable_incidence_matrix == repeat(
            variable_indices_flattened, 
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            ENC_IN=ENC_IN
        )).to(torch.float32)
        variable_incidence_matrix = variable_incidence_matrix * repeat(
            x_y_mask_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            ENC_IN=ENC_IN
        ) # remove non-observed values at tail

        # init observation nodes (BATCH_SIZE, N_OBSERVATIONS_MAX, 2) -> (BATCH_SIZE, N_OBSERVATIONS_MAX, d_model)
        observation_nodes = self.relu(self.observation_node_encoder(x_L_flattened)) * repeat(
            x_y_mask_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
            D=self.d_model
        )
        # init temporal hyperedges (BATCH_SIZE, L, d_model) 
        temporal_hyperedges = torch.sin(self.temporal_hyperedge_encoder(x_y_mark))  
        # init variable hyperedges (BATCH_SIZE, ENC_IN, d_model) 
        variable_hyperedges = self.relu(
            repeat(
                self.variable_hyperedge_weights.to(x_L_flattened.device), 
                "ENC_IN D_MODEL -> BATCH_SIZE ENC_IN D_MODEL",
                BATCH_SIZE=BATCH_SIZE
            )
        )

        return (
            observation_nodes, 
            temporal_hyperedges, 
            variable_hyperedges,
            temporal_incidence_matrix,
            variable_incidence_matrix
        )
            

class HypergraphLearner(nn.Module):
    '''
    Message passing:

    - nodes-to-hyperedge
    - hyperedge-to-hyperedge
    - hyperedge-to-node

    implemented via attention
    '''
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        time_length: int,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model

        self.activation = nn.ReLU()

        self.node2variable_hyperedge = nn.ModuleList(
            MultiHeadAttentionBlock(
                dim_Q=d_model, 
                dim_K=2*d_model, 
                dim_V=2*d_model,
                n_dim=d_model, 
                num_heads=n_heads
            ) for _ in range(self.n_layers)
        )
        self.node2temporal_hyperedge = nn.ModuleList(
            MultiHeadAttentionBlock(
                dim_Q=d_model, 
                dim_K=2*d_model, 
                dim_V=2*d_model,
                n_dim=d_model, 
                num_heads=n_heads
            ) for _ in range(self.n_layers)
        )
        self.node_self_update = nn.ModuleList(
            MultiHeadAttentionBlock(
                dim_Q=d_model, 
                dim_K=3*d_model, 
                dim_V=3*d_model,
                n_dim=d_model, 
                num_heads=n_heads
            ) for _ in range(self.n_layers)
        )
        self.variable_hyperedge2variable_hyperedge = IrregularityAwareAttention(
            d_model=d_model
        )
        self.hyperedge2node = nn.ModuleList(
            nn.Linear(3*d_model, d_model) for _ in range(self.n_layers)
        )

        self.hyperedge2hyperedge_layers = [
            n_layers - 1
        ]
        self.scale = 1 / time_length

    def forward(
        self,
        observation_nodes: Tensor, 
        temporal_hyperedges: Tensor, 
        variable_hyperedges: Tensor,
        time_indices_flattened: Tensor,
        variable_indices_flattened: Tensor,
        temporal_incidence_matrix: Tensor, 
        variable_incidence_matrix: Tensor,
        x_y_mask_flattened: Tensor,
        x_y_mask: Tensor,
        y_mask_L_flattened: Tensor
    ):
        for i in range(self.n_layers):
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B L N", L=temporal_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8
            # observation nodes to temporal hyperedges
            temporal_hyperedges_updated = self.node2temporal_hyperedge[i](
                temporal_hyperedges, 
                torch.cat([
                    variable_hyperedges.gather(dim=1, index=repeat(
                        variable_indices_flattened,
                        "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                        D=self.d_model
                    )),
                    observation_nodes
                ], -1), 
                temporal_incidence_matrix if i != 0 else temporal_incidence_matrix * mask_temp
            )    
            if i == 0:
                mask_temp = 1 - repeat(y_mask_L_flattened, "B N -> B ENC_IN N", ENC_IN=variable_incidence_matrix.shape[1])
                mask_temp[mask_temp == 0] = 1e-8
            # observation nodes to variable hyperedges
            variable_hyperedges_updated = self.node2variable_hyperedge[i](
                variable_hyperedges, 
                torch.cat([
                    temporal_hyperedges.gather(dim=1, index=repeat(
                        time_indices_flattened,
                        "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                        D=self.d_model
                    )), 
                    observation_nodes
                ], -1), 
                variable_incidence_matrix if i != 0 else variable_incidence_matrix * mask_temp
            )
            variable_hyperedges = variable_hyperedges_updated
            temporal_hyperedges = temporal_hyperedges_updated

            # hyperedge-to-node update
            temporal_hyperedges_gathered = temporal_hyperedges.gather(
                dim=1, 
                index=repeat(
                    time_indices_flattened,
                    "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                    D=self.d_model
                )
            )
            variable_hyperedges_gathered = variable_hyperedges.gather(
                dim=1, 
                index=repeat(
                    variable_indices_flattened,
                    "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                    D=self.d_model
                )
            ) # (BATCH_SIZE, N_OBSERVATIONS_MAX, d_model)
            observation_nodes_updated = self.node_self_update[i](
                observation_nodes, 
                torch.cat([temporal_hyperedges_gathered, variable_hyperedges_gathered, observation_nodes], -1), 
                x_y_mask_flattened.unsqueeze(2) * x_y_mask_flattened.unsqueeze(1)
            )
            observation_nodes = self.activation(
                (observation_nodes + self.hyperedge2node[i](torch.cat([observation_nodes_updated, temporal_hyperedges_gathered, variable_hyperedges_gathered], dim=-1))) * \
                repeat(
                    x_y_mask_flattened,
                    "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE N_OBSERVATIONS_MAX D",
                    D=self.d_model
                )
            )

            if i in self.hyperedge2hyperedge_layers:
                # perform hyperedge-to-hyperedge message passing after fully learned in previous layers
                sync_mask = x_y_mask
                query_and_key = self.get_fine_grained_embedding(observation_nodes, sync_mask)
                merge_coefficients = sync_mask.transpose(-1, -2) @ sync_mask
                n_observations_per_variable = merge_coefficients.diagonal(offset=0, dim1=-2, dim2=-1)
                merge_coefficients[n_observations_per_variable != 0] = (merge_coefficients / repeat(n_observations_per_variable, "B ENC_IN -> B ENC_IN ENC_IN_2", ENC_IN_2=sync_mask.shape[-1]))[n_observations_per_variable != 0]
                variable_hyperedges_updated = variable_hyperedges_updated + self.variable_hyperedge2variable_hyperedge(
                    x=variable_hyperedges_updated,
                    query_aux=query_and_key,
                    key_aux=query_and_key,
                    merge_coefficients=merge_coefficients
                )
                variable_hyperedges = variable_hyperedges_updated

        return (
            observation_nodes,
            temporal_hyperedges,
            variable_hyperedges,
        )

    def get_fine_grained_embedding(
        self, 
        tensor_flattened: Tensor, 
        target_shape: Tensor, 
    ):
        BATCH_SIZE, L, ENC_IN = target_shape.shape
        D_MODEL = tensor_flattened.shape[-1]
        tensor_flattened = tensor_flattened[:, :, :max(1, int(D_MODEL * self.scale))]
        D_MODEL_SCALED = tensor_flattened.shape[-1]
        result = torch.zeros((BATCH_SIZE, L, ENC_IN, D_MODEL_SCALED), dtype=tensor_flattened.dtype, device=tensor_flattened.device)

        for i in range(BATCH_SIZE):
            masked_indices = target_shape[i].unsqueeze(-1).repeat(1, 1, 1, D_MODEL_SCALED).view(-1).nonzero(as_tuple=True)[0]
            tensor_flattened_viewed = tensor_flattened.reshape(BATCH_SIZE, -1)[i][:len(masked_indices)]
            result[i].view(-1)[masked_indices] = tensor_flattened_viewed

        return rearrange(
            result,
            "B L ENC_IN D_MODEL_SCALED -> B ENC_IN (L D_MODEL_SCALED)"
        )

class MultiHeadAttentionBlock(nn.Module):
    '''
    adapted from GraFITi
    '''
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super(MultiHeadAttentionBlock, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.n_dim = n_dim
        self.fc_q = nn.Linear(dim_Q, n_dim)
        self.fc_k = nn.Linear(dim_K, n_dim)
        self.fc_v = nn.Linear(dim_K, n_dim)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(
        self, 
        Q: Tensor, 
        K: Tensor, 
        mask: Tensor = None
    ):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)
        K = torch.cat(K.split(dim_split, dim=2), dim=0)
        V = torch.cat(V.split(dim_split, dim=2), dim=0)

        Att_mat = Q_.bmm(K.transpose(1, 2))/math.sqrt(self.n_dim) # (B * num_heads, enc_in, n_observations_max)

        if mask is not None:
            Att_mat = Att_mat.masked_fill(
                mask.repeat(self.num_heads, 1, 1) == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), dim=0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class IrregularityAwareAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        self.scale = d_model ** 0.5

        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    
    def forward(
        self, 
        x: Tensor, 
        query_aux: Tensor = None,
        key_aux: Tensor = None,
        adjacency_mask: Tensor = None,
        merge_coefficients: Tensor = None,
    ):
        batch_size, n_variables, hidden_dim = x.shape
        
        query = self.query_proj(x) 
        key = self.key_proj(x)     
        value = self.value_proj(x) 
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        mask_value = torch.finfo(attention_scores.dtype).min

        if query_aux is not None and key_aux is not None:
            query_aux = query_aux
            key_aux = key_aux

            attention_scores_aux = torch.matmul(query_aux, key_aux.transpose(-2, -1)) / (key_aux.shape[-1] ** 0.5)
            
            non_zero_mask = (attention_scores_aux != 0)
            positive_mask = (attention_scores > self.threshold)

            mask = positive_mask & non_zero_mask

            attention_scores[mask]  = ((1 - merge_coefficients) * attention_scores + merge_coefficients * attention_scores_aux)[mask]
        
        # If adjacency mask is provided, apply it to attention scores
        if adjacency_mask is not None:
            assert adjacency_mask.shape == (batch_size, n_variables, n_variables), \
                f"Adjacency mask shape must be {(batch_size, n_variables, n_variables)}, " \
                f"got {adjacency_mask.shape}"
            
            masked_attention_scores = attention_scores.masked_fill(
                adjacency_mask == 0, mask_value
            )
        else:
            masked_attention_scores = attention_scores
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(masked_attention_scores, dim=-1) # (batch_size, n_variables, n_variables)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)

        return attended_values