import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Model(nn.Module):
    '''
    - paper: "Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach" (ICML 2024)
    - paper link: https://openreview.net/forum?id=UZlMXUGI6e
    - code adapted from: https://github.com/usail-hkust/t-PatchGNN
    '''
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.task_name = configs.task_name
        self.configs = configs
        self.hid_dim = configs.d_model # 64
        self.ndim = configs.enc_in

        assert configs.seq_len % configs.patch_len == 0, f"seq_len {configs.seq_len} should be divisible by patch_len {configs.patch_len}"
        self.n_patch: int = configs.seq_len // configs.patch_len

        self.batch_size = configs.batch_size
        self.supports = None
        self.n_layer = configs.n_layers # 1
        dropout = configs.dropout # 0
        self.te_dim = configs.tpatchgnn_te_dim
        self.n_heads = configs.n_heads
        self.tf_layer = 1
        self.node_dim = configs.node_dim
        self.hop = 1
        self.outlayer = "Linear"

        ### Intra-time series modeling ## 
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        ## TTCN
        input_dim = 1 + self.te_dim
        ttcn_dim = self.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.Filter_Generators = nn.Sequential(
                nn.Linear(input_dim, ttcn_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(ttcn_dim, ttcn_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
        
        d_model = self.hid_dim
        ## Transformer
        self.ADD_PE = PositionalEncoding(d_model) 
        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.n_heads, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=self.tf_layer))			

        ### Inter-time series modeling ###
        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        nodevec_dim = self.node_dim
        self.nodevec_dim = nodevec_dim
        if self.supports is None:
            self.supports = []

        self.nodevec1 = nn.Parameter(torch.randn(self.ndim, nodevec_dim).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, self.ndim).cuda(), requires_grad=True)

        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.n_layer):
            self.nodevec_linear1.append(nn.Linear(self.hid_dim, nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(self.hid_dim, nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
                nn.Linear(self.hid_dim+nodevec_dim, 1),
                nn.Tanh(),
                nn.ReLU()))
            self.nodevec_gate2.append(nn.Sequential(
                nn.Linear(self.hid_dim+nodevec_dim, 1),
                nn.Tanh(),
                nn.ReLU()))
            
        self.supports_len +=1

        self.gconv = nn.ModuleList() # gragh conv
        for _ in range(self.n_layer):
            self.gconv.append(gcn(d_model, d_model, dropout, support_len=self.supports_len, order=self.hop))

        ### Encoder output layer ###
        enc_dim = self.hid_dim
        if(self.outlayer == "Linear"):
            self.temporal_agg = nn.Sequential(
                nn.Linear(self.hid_dim*self.n_patch, enc_dim))
        
        elif(self.outlayer == "CNN"):
            self.temporal_agg = nn.Sequential(
                nn.Conv1d(d_model, enc_dim, kernel_size=self.n_patch))

        ### Decoder ###
        if configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            self.decoder = nn.Sequential(
                nn.Linear(enc_dim+self.te_dim, self.hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hid_dim, 1)
            )
        else:
            raise NotImplementedError
        
    def LearnableTE(self, tt):
        # tt: (N*M*B, L, 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
    
    def TTCN(self, X_int, mask_X):
        # X_int: shape (B*N*M, L, F)
        # mask_X: shape (B*N*M, L, 1)

        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
        return h_t

    def IMTS_Model(self, x, mask_X):
        """
        x (B*N*M, L, F)
        mask_X (B*N*M, L, 1)
        """
        # mask for the patch
        mask_patch = (mask_X.sum(dim=1) > 0) # (B*N*M, 1)

        ### TTCN for patch modeling ###
        x_patch = self.TTCN(x, mask_X) # (B*N*M, hid_dim-1)
        x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim)
        x_patch = x_patch.view(self.batch_size, self.ndim, self.n_patch, -1) # (B, N, M, hid_dim)
        B, N, M, D = x_patch.shape

        x = x_patch
        for layer in range(self.n_layer):

            if(layer > 0): # residual
                x_last = x.clone()
                
            ### Transformer for temporal modeling ###
            x = x.reshape(B*N, M, -1) # (B*N, M, F)
            x = self.ADD_PE(x)
            x = self.transformer_encoder[layer](x).view(x_patch.shape) # (B, N, M, F)

            ### GNN for inter-time series modeling ###
            ### time-adaptive graph structure learning ###
            nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            x_p1 = x_gate1 * self.nodevec_linear1[layer](x) # (B, M, N, 10)
            x_p2 = x_gate2 * self.nodevec_linear2[layer](x) # (B, M, N, 10)
            nodevec1 = nodevec1 + x_p1.permute(0,2,1,3) # (B, M, N, 10)
            nodevec2 = nodevec2 + x_p2.permute(0,2,3,1) # (B, M, 10, N)

            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1) # (B, M, N, N) used
            new_supports = self.supports + [adp]

            # input x shape (B, F, N, M)
            x = self.gconv[layer](x.permute(0,3,1,2), new_supports) # (B, F, N, M)
            x = x.permute(0, 2, 3, 1) # (B, N, M, F)

            if(layer > 0): # residual addition
                x = x_last + x 

        ### Output layer ###
        if(self.outlayer == "CNN"):
            x = x.reshape(self.batch_size*self.ndim, self.n_patch, -1).permute(0, 2, 1) # (B*N, F, M)
            x = self.temporal_agg(x) # (B*N, F, M) -> (B*N, F, 1)
            x = x.view(self.batch_size, self.ndim, -1) # (B, N, F)

        elif(self.outlayer == "Linear"):
            x = x.reshape(self.batch_size, self.ndim, -1) # (B, N, M*F)
            x = self.temporal_agg(x) # (B, N, hid_dim)

        return x

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mark: Tensor = None, 
        y_mask: Tensor = None, 
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
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning(f"y is missing for the model input. This is only reasonable when the model is testing flops!")
            y = torch.ones((BATCH_SIZE, Y_LEN, ENC_IN), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(end=y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "L -> B L 1", B=y.shape[0])
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        # reshape tensors to align with tPatchGNN's desired input shape
        # x_enc: [B, npatch, seq_len // npatch, enc_in]
        # x_mark_enc: [B, npatch, seq_len // npatch, enc_in]
        # x_mark_dec: [B, pred_len]
        # x_mask_enc: [B, npatch, seq_len // npatch, enc_in]
        x_mark = repeat(x_mark[:, :, 0], "b l -> b l f", f=x.shape[-1])
        y_mark = repeat(y_mark[:, :, 0], "b l -> b l f", f=y.shape[-1])
        B, seq_len, enc_in = x_mark.shape
        B, pred_len, c_out = y_mark.shape
        if seq_len % self.n_patch != 0:
            logger.exception(f"Error in tPatchGNN forward(): seq_len must be divisible by number of patches {self.n_patch}. Try changing the self.npatch value.", stack_info=True)
            exit(1)
        x = x.reshape(B, self.n_patch, seq_len // self.n_patch, -1)
        x_mark = x_mark.reshape(B, self.n_patch, seq_len // self.n_patch, -1)
        x_mask = x_mask.reshape(B, self.n_patch, seq_len // self.n_patch, -1)
        y_mark = y_mark[:, :, 0]
        # END adaptor

        # original codes below

        B, npatch, seq_len_in_patch, enc_in = x.shape
        x = x.permute(0, 3, 1, 2).reshape(-1, seq_len_in_patch, 1) # (B * enc_in * npatch, seq_len_in_patch, 1)
        x_mark = x_mark.permute(0, 3, 1, 2).reshape(-1, seq_len_in_patch, 1)  # (B * enc_in * npatch, seq_len_in_patch, 1)
        x_mask = x_mask.permute(0, 3, 1, 2).reshape(-1, seq_len_in_patch, 1)  # (B * enc_in * npatch, seq_len_in_patch, 1)
        te_his = self.LearnableTE(x_mark) # (B * enc_in * npatch, seq_len_in_patch, F_te)

        x = torch.cat([x, te_his], dim=-1)  # (B * enc_in * npatch, seq_len_in_patch, F)

        ### *** an encoder to model irregular time series
        h = self.IMTS_Model(x, x_mask) # (B, enc_in, hid_dim)


        if self.configs.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            """ Decoder """
            L_pred = y_mark.shape[-1]
            h = h.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1) # (B, enc_in, Lp, F)
            y_mark = y_mark.view(B, 1, L_pred, 1).repeat(1, enc_in, 1, 1) # (B, enc_in, Lp, 1)
            te_pred = self.LearnableTE(y_mark) # (B, enc_in, Lp, F_te)

            h = torch.cat([h, te_pred], dim=-1) # (B, enc_in, Lp, F)

            # (B, enc_in, Lp, F) -> (B, enc_in, Lp, 1) -> (B, Lp, enc_in)
            outputs = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1) 
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": outputs[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        else:
            raise NotImplementedError


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        # x (B, F, N, M)
        # A (B, M, N, N)
        x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
        # print(x.shape)
        return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear,self).__init__()
        # self.mlp = nn.Linear(c_in, c_out)
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        # x (B, F, N, M)

        # return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.mlp(x)
        
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        # c_in = (order*support_len)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x (B, F, N, M)
        # a (B, M, N, N)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1) # concat x and x_conv
        h = self.mlp(h)
        return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

