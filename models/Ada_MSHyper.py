import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, softmax
from einops import *

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from layers.Ada_MSHyper.Layers import *


class Model(nn.Module):
    """
    - paper: "Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting" (NeurIPS 2024)
    - paper link: https://openreview.net/forum?id=RNbrIQ0se8
    - code adapted from: https://github.com/shangzongjiang/Ada-MSHyper
    """

    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.window_size = [4, 4]

        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        # 以下为超图设计代码
        self.all_size = get_mask(self.seq_len, self.window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = Bottleneck_Construct(
            configs.enc_in, self.window_size, configs.enc_in
        )
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight = nn.Parameter(
            (1 / self.Ms_length) * torch.ones([self.pred_len, self.Ms_length])
        )
        self.chan_tran = nn.Linear(configs.d_model, configs.enc_in)
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tra = nn.Linear(320, self.pred_len)

        ###以下为embedding实现
        self.dim = configs.d_model
        self.hyper_num = 50
        self.embedhy = nn.Embedding(self.hyper_num, self.dim)
        self.embednod = nn.Embedding(self.Ms_length, self.dim)

        self.idx = torch.arange(self.hyper_num)
        self.nodidx = torch.arange(self.Ms_length)
        self.alpha = 3
        self.k = 10

        self.window_size = [4, 4]
        self.multi_adaptive_hypergraph = multi_adaptive_hypergraph(configs)
        self.hyper_num1 = [50, 20, 10]
        self.hyconv = nn.ModuleList()
        self.hyperedge_atten = SelfAttentionLayer(configs)
        for i in range(len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))

        self.slicetran = nn.Linear(100, self.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.argg.append(nn.Linear(self.all_size[i], self.pred_len))
        self.chan_tran = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(
        self, 
        x: Tensor, 
        y: Tensor = None, 
        y_mask: Tensor = None, 
        **kwargs
    ):
        # BEGIN adaptor
        BATCH_SIZE, SEQ_LEN, ENC_IN = x.shape
        Y_LEN = self.pred_len
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones_like(x, dtype=x.dtype, device=x.device)
        if y_mask is None:
            y_mask = torch.ones_like(y, dtype=y.dtype, device=y.device)
        # END adaptor

        # normalization
        mean_enc = x.mean(1, keepdim=True).detach()
        x = x - mean_enc
        std_enc = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / std_enc

        adj_matrix = self.multi_adaptive_hypergraph(x)

        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        result_tensor1 = []
        for i in range(len(self.hyper_num1)):
            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###尺度间关系
            node_value = seq_enc[i].permute(0, 2, 1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums = {}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id = edge_id.item()
                    node_id = node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]

            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)

            ###尺度内关系
            output, constrainloss = self.hyconv[i](seq_enc[i], mask)
            result_tensor1.append(self.argg[i](seq_enc[i].permute(0, 2, 1)))

            if i == 0:
                result_tensor = output
                result_conloss = constrainloss
            else:
                result_tensor = torch.cat((result_tensor, output), dim=0)
                result_conloss += constrainloss

        result_tensor = rearrange(
            result_tensor, "Z BATCH_SIZE ENC_IN -> BATCH_SIZE Z ENC_IN"
        )  # Z's meaning to be determined

        result_tensor1 = sum(result_tensor1) / len(self.hyper_num1)

        sum_hyper_list = torch.cat(sum_hyper_list, dim=1)
        sum_hyper_list = sum_hyper_list.to(x.device)
        padding_need = 80 - sum_hyper_list.size(1)
        hyperedge_attention = self.hyperedge_atten(sum_hyper_list)
        pad = torch.nn.functional.pad(
            hyperedge_attention, (0, 0, 0, padding_need, 0, 0)
        )
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            if self.individual:
                output = torch.zeros(
                    [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
                ).to(x.device)
                for i in range(self.channels):
                    output[:, :, i] = self.Linear[i](x[:, :, i])
                x = output
            else:
                x = self.Linear(x.permute(0, 2, 1))
                x_out = self.out_tran(result_tensor.permute(0, 2, 1))  ###ori
                x_out_inter = self.inter_tran(pad.permute(0, 2, 1))

            x = x_out + x + x_out_inter
            x = self.Linear_Tran(x).permute(0, 2, 1)
            x = x * std_enc + mean_enc
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": x[:, -PRED_LEN:, f_dim:], 
                "true": y[:, :, f_dim:], 
                "mask": y_mask[:, :, f_dim:],
                "loss2": result_conloss
            }
        else:
            raise NotImplementedError


class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=True,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.1,
        bias=False,
    ):
        super(HypergraphConv, self).__init__(aggr="add")
        self.soft = nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # 初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    ####message在消息传递中计算每个节点收到的消息
    #####将输入的节点特征和超边归一化权重相乘
    ####并根据头数和输出通道数将结果重新组织
    def message(self, x_j, edge_index_i, norm, alpha):
        """
        - x_j: [SEQ_LEN, BATCH_SIZE, ENC_IN] Features of the neighboring nodes.
        - edge_index_i: [SEQ_LEN] Indices of the incoming edges for the current node.
        - norm: [22]? Normalization factors for the edges.
        - alpha: [SEQ_LEN, BATCH_SIZE] Attention weights (if applicable).
        """
        # print(f"{x_j.shape=}")
        # print(f"{norm.shape=}")
        # print(f"{edge_index_i.shape=}")
        out = norm[edge_index_i].view(1, -1, 1) * x_j  ####
        # print(f"{out.shape=}")
        # print(f"{alpha.shape=}")
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        """
        - hyperedge_index: [2, SEQ_LEN]
        """
        x = torch.matmul(x, self.weight)  # (BATCH_SIZE, SEQ_LEN, ENC_IN)
        x = rearrange(
            x, "BATCH_SIZE SEQ_LEN ENC_IN -> SEQ_LEN BATCH_SIZE ENC_IN"
        )  # (SEQ_LEN, BATCH_SIZE, ENC_IN)

        # Selects the features of the source nodes for the hyperedges.
        x_i = torch.index_select(
            x, dim=0, index=hyperedge_index[0]
        )  # (SEQ_LEN, BATCH_SIZE, ENC_IN)

        # Sums the features of the nodes for each hyperedge, storing results in a dictionary.
        edge_sums = {}  # items: 22. Each item: [BATCH_SIZE, ENC_IN]
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x[node_id, :, :]
            else:
                edge_sums[edge_id] += x[node_id, :, :]
        result_list = torch.stack(
            [value for value in edge_sums.values()], dim=0
        )  # (22?, BATCH_SIZE, ENC_IN)
        x_j = torch.index_select(
            result_list, dim=0, index=hyperedge_index[1]
        )  # (SEQ_LEN, BATCH_SIZE, ENC_IN)
        loss_hyper = 0

        # Calculates the inner product and norms of the edge features.
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(
                    edge_sums[k] * edge_sums[m], dim=1, keepdim=True
                )  # (BATCH_SIZE, 1)
                norm_q_i = torch.norm(
                    edge_sums[k], dim=1, keepdim=True
                )  # (BATCH_SIZE, 1)
                norm_q_j = torch.norm(
                    edge_sums[m], dim=1, keepdim=True
                )  # (BATCH_SIZE, 1)
                alpha = inner_product / (norm_q_i * norm_q_j)

                distan = torch.norm(
                    edge_sums[k] - edge_sums[m], dim=1, keepdim=True
                )  # (BATCH_SIZE, 1)

                loss_item = alpha * distan + (1 - alpha) * (
                    torch.clamp(torch.tensor(4.2) - distan, min=0.0)
                )  # (BATCH_SIZE, 1)
                loss_hyper += torch.abs(torch.mean(loss_item))

        loss_hyper = loss_hyper / ((len(edge_sums) + 1) ** 2)

        # Combines the features of connected nodes to compute attention weights
        # Leaky ReLU and softmax for normalization
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # [1008,1]
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))  # [1008,1]
        alpha = F.dropout(
            alpha, p=self.dropout, training=self.training
        )  # (SEQ_LEN, BATCH_SIZE)

        # Computes the degree of nodes and prepares normalization factors for propagation
        D = degree(hyperedge_index[0], x.size(0), x.dtype)  # (SEQ_LEN)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)  # (44)?
        B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)  # (22)?
        B[B == float("inf")] = 0

        # Message passing
        self.flow = "source_to_target"
        # print(f"{x1.permute(1, 0, 2).shape=}")
        out = self.propagate(
            hyperedge_index, x=x.permute(1, 0, 2), norm=B, alpha=alpha.permute(1, 0)
        )
        # print(f"{out.shape=}")
        self.flow = "target_to_source"
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha.permute(1, 0))
        out = out.transpose(0, 1)

        constrain_loss_total = abs(torch.mean(x_i - x_j)) + loss_hyper

        return out, constrain_loss_total

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class multi_adaptive_hypergraph(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(multi_adaptive_hypergraph, self).__init__()
        self.seq_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        self.window_size = [4, 4]
        self.inner_size = 5
        self.dim = configs.d_model
        self.hyper_num = [50, 20, 10]
        self.alpha = 3
        self.k = 3
        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        self.linhy = nn.ModuleList()
        self.linnod = nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i], self.dim))
            self.linhy.append(nn.Linear(self.dim, self.dim))
            self.linnod.append(nn.Linear(self.dim, self.dim))
            if i == 0:
                self.embednod.append(nn.Embedding(self.seq_len, self.dim))
            else:
                product = math.prod(self.window_size[:i])
                layer_size = math.floor(self.seq_len / product)
                self.embednod.append(nn.Embedding(int(layer_size), self.dim))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        node_num = []
        node_num.append(self.seq_len)
        # window_size[4,4],node_num变为[336,84,21]
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        hyperedge_all = []
        node_all = []

        # 每个尺度的超边数量是超参[50,20,10]
        for i in range(len(self.hyper_num)):
            hypidxc = torch.arange(self.hyper_num[i]).to(x.device)

            nodeidx = torch.arange(node_num[i]).to(x.device)

            hyperen = self.embedhy[i](hypidxc)
            nodeec = self.embednod[i](nodeidx)
            # 生成点边关联矩阵
            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj = F.softmax(F.relu(self.alpha * a))

            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float("0"))
            s1, t1 = adj.topk(min(adj.size(1), self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(
                adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device)
            )
            # 去掉全为0的列
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = torch.tensor(adj, dtype=torch.int)
            result_list = [
                list(torch.nonzero(matrix_array[:, col]).flatten().tolist())
                for col in range(matrix_array.shape[1])
            ]
            ##假设有四个节点，三条超边，则最终形成的矩阵形似如下,其中上面是节点集合，下面是超边集合
            # [1,2,3,1,2,4,2,3,4]
            # [1,1,1,2,2,2,3,3,3]
            node_list = torch.cat(
                [torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]
            ).tolist()
            count_list = list(torch.sum(adj, dim=0).tolist())
            hperedge_list = torch.cat(
                [
                    torch.full((count,), idx)
                    for idx, count in enumerate(count_list, start=0)
                ]
            ).tolist()

            hypergraph = np.vstack((node_list, hperedge_list))
            hyperedge_all.append(hypergraph)

        a = hyperedge_all
        return a


class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)

        # 计算 attention 分数
        attention_scores = F.softmax(
            torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1
        )

        # 使用 attention 分数加权平均值
        attended_values = torch.matmul(attention_scores, v)

        return attended_values


def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size
