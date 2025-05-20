import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import *

class MAB2(nn.Module):
    '''
    Multi-head Attention Block
    '''
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False):
        super(MAB2, self).__init__()
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

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.n_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K = torch.cat(K.split(dim_split, 2), 0)
        V = torch.cat(V.split(dim_split, 2), 0)

        Att_mat = Q_.bmm(K.transpose(1, 2))/math.sqrt(self.n_dim)
        if mask is not None:
            Att_mat = Att_mat.masked_fill(
                mask.repeat(self.num_heads, 1, 1) == 0, -10e9)
        A = torch.softmax(Att_mat, 2)
        O = torch.cat((Q_ + A.bmm(V)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class Encoder(nn.Module):
    def __init__(self, dim=41, nkernel=128, n_layers=3, attn_head=4, task_name="short_term_forecast", n_classes=2):
        super(Encoder, self).__init__()
        self.task_name = task_name
        self.dim = dim+2
        self.nheads = attn_head
        self.nkernel = nkernel
        self.edge_init = nn.Linear(2, nkernel)
        self.chan_init = nn.Linear(dim, nkernel)
        self.time_init = nn.Linear(1, nkernel)
        self.n_layers = n_layers
        self.channel_time_attn = nn.ModuleList()
        self.time_channel_attn = nn.ModuleList()
        self.edge_nn = nn.ModuleList()
        self.output = nn.Linear(3*nkernel, 1)
        for i in range(self.n_layers):
            self.channel_time_attn.append(
                MAB2(nkernel, 2*nkernel, 2*nkernel, nkernel, self.nheads))
            self.time_channel_attn.append(
                MAB2(nkernel, 2*nkernel, 2*nkernel, nkernel, self.nheads))
            self.edge_nn.append(nn.Linear(3*nkernel, nkernel))
        self.relu = nn.ReLU()


    def gather(self, x: torch.Tensor, inds: torch.Tensor):
        # inds =  # keep repeating until the embedding len as a new dim
        return x.gather(dim=1, index=inds[:, :, None].repeat(1, 1, x.shape[-1]))

    def forward(self, context_x, value, mask, target_value, target_mask, exp_stage):
        """
        - context_x: x_mark + y_mark
        - value: x + zero padding
        - mask: x_mask + y_mask
        - target_value: zero padding + y
        - target_mask: zero padding + y_mask
        """
        BATCH_SIZE, L, ENC_IN = value.shape  # C
        x_y_mark = context_x[:, :, None]  # (BATCH_SIZE, L) -> (BATCH_SIZE, L, 1)
        time_indices = torch.cumsum(torch.ones_like(value).to(torch.int64), dim=1) - 1  # (BATCH_SIZE, L, ENC_IN) init for time indices. 0, 1, 2...
        channel_indices = torch.cumsum(torch.ones_like(value).to(torch.int64), dim=-1) - 1  # (BATCH_SIZE, L, ENC_IN) init for channel indices. 0, 1, 2...
        mask_bool = mask.to(torch.bool)  # (BATCH_SIZE, L, ENC_IN)

        # get total number of observations in each sample, and take max
        N_OBSERVATIONS_MAX = torch.max(mask.sum((1, 2))).to(torch.int64)  # flattened TxC max length possible

        # function to pad 0s to the last dimension of input v, with no padding on left side and some paddings on right side
        def pad(v): return F.pad(v, [0, N_OBSERVATIONS_MAX - len(v)], value=0)

        # flatten everything, from (L, ENC_IN) to (N_OBSERVATIONS_MAX), where observations belonging to the same timestep are nearby
        # note that r[m] won't keep the original tensor shape by default, thus flattened
        value_flattened = torch.stack([pad(r[m]) for r, m in zip(value, mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        time_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(time_indices, mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        channel_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(channel_indices, mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        mask_flattened = torch.stack([pad(r[m]) for r, m in zip(mask, mask_bool)]).contiguous() # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)

        target_value_flattened = torch.stack([pad(r[m]) for r, m in zip(target_value, mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)
        target_mask_flattened = torch.stack([pad(r[m]) for r, m in zip(target_mask, mask_bool)]).contiguous()  # (BATCH_SIZE, L, ENC_IN) -> (BATCH_SIZE, N_OBSERVATIONS_MAX)


        channel_IDs = torch.ones([BATCH_SIZE, ENC_IN]).cumsum(dim=1).to(context_x.device) - 1  # (BATCH_SIZE, ENC_IN) prepare for later one hot encoding channels. 0, 1, 2,...
        channel_embedding = torch.nn.functional.one_hot(
            channel_IDs.to(torch.int64), 
            num_classes=ENC_IN
        ).to(torch.float32)  # (BATCH_SIZE, ENC_IN, ENC_IN) #channel one hot encoding

        # unusually, lookback window are 0s
        lookback_mask_flattened = 1 - mask_flattened + target_mask_flattened # (BATCH_SIZE, N_OBSERVATIONS_MAX)
        # lookback_mask_flattened = mask_flattened - target_mask_flattened # (BATCH_SIZE, N_OBSERVATIONS_MAX)
        value_flattened = torch.cat([value_flattened[:, :, None], lookback_mask_flattened[:, :, None]], -1) # (BATCH_SIZE, N_OBSERVATIONS_MAX) -> (BATCH_SIZE, N_OBSERVATIONS_MAX, 2)

        # channel mask (BATCH_SIZE, ENC_IN, N_OBSERVATIONS_MAX)
        # indicate for every variable, which observation belongs to it
        channel_mask = repeat(
            channel_IDs, 
            "BATCH_SIZE ENC_IN -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            N_OBSERVATIONS_MAX=N_OBSERVATIONS_MAX
        )
        temp_c_inds = repeat(
            channel_indices_flattened, 
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            ENC_IN=ENC_IN
        )
        channel_mask = (channel_mask == temp_c_inds).to(torch.float32)
        del temp_c_inds
        channel_mask = channel_mask * repeat(
            mask_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE ENC_IN N_OBSERVATIONS_MAX", 
            ENC_IN=ENC_IN
        ) # remove non-observed values at tail

        # time mask (BATCH_SIZE, L, N_OBSERVATIONS_MAX)
        # indicate for every timestep, which observation belongs to it
        time_mask = repeat(
            time_indices_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE L N_OBSERVATIONS_MAX",
            L=L
        )
        temp_T_inds = torch.ones_like(x_y_mark[:, :, 0]).cumsum(dim=1)[:, :, None].repeat(1, 1, N_OBSERVATIONS_MAX) - 1
        time_mask = (time_mask == temp_T_inds).to(torch.float32)  # BxTxK_
        del temp_T_inds
        time_mask = time_mask * repeat(
            mask_flattened,
            "BATCH_SIZE N_OBSERVATIONS_MAX -> BATCH_SIZE L N_OBSERVATIONS_MAX", 
            L=L
        ) # remove non-observed values at tail

        # (BATCH_SIZE, N_OBSERVATIONS_MAX, 2) -> (BATCH_SIZE, N_OBSERVATIONS_MAX, nkernel)
        value_flattened = self.relu(self.edge_init(value_flattened)) * \
            mask_flattened[:, :, None].repeat(1, 1, self.nkernel)
        # learned time embedding (BATCH_SIZE, L, nkernel) 
        time_embedding = torch.sin(self.time_init(x_y_mark))  
        # embedding on one-hot encoded channel (BATCH_SIZE, ENC_IN, nkernel) 
        channel_embedding = self.relu(self.chan_init(channel_embedding))


        for i in range(self.n_layers):

            # channels as queries
            q_c = channel_embedding # (BATCH_SIZE, ENC_IN, nkernel) 
            k_t = self.gather(time_embedding, time_indices_flattened) # (BATCH_SIZE, N_OBSERVATIONS_MAX, nkernel)
            k = torch.cat([k_t, value_flattened], -1) # (BATCH_SIZE, N_OBSERVATIONS_MAX, 2*nkernel)

            # attn (channel_embd, concat(time, values)) along with the mask
            C__ = self.channel_time_attn[i](q_c, k, channel_mask)

            # times as queries
            q_t = time_embedding
            k_c = self.gather(channel_embedding, channel_indices_flattened) # (BATCH_SIZE, N_OBSERVATIONS_MAX, nkernel)
            k = torch.cat([k_c, value_flattened], -1) # (BATCH_SIZE, N_OBSERVATIONS_MAX, 2*nkernel)
            T__ = self.time_channel_attn[i](q_t, k, time_mask)

            # updating edge weights
            value_flattened = self.relu(value_flattened + self.edge_nn[i](torch.cat([value_flattened, k_t, k_c], dim=-1))) * mask_flattened[:, :, None].repeat(1, 1, self.nkernel)

            # updating only channel nodes

            channel_embedding = C__
            time_embedding = T__

        if self.task_name in ["short_term_forecast", "long_term_forecast", "imputation"]:
            k_t = self.gather(time_embedding, time_indices_flattened)
            k_c = self.gather(channel_embedding, channel_indices_flattened)
            output = self.output(torch.cat([value_flattened, k_t, k_c], dim=-1))
            return output, target_value_flattened, target_mask_flattened
        else:
            raise NotImplementedError
