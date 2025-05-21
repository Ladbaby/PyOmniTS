import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from layers.Warpformer.Layers import EncoderLayer
from layers.Warpformer.Modules import Attention
from layers.Warpformer.WarpingLayer import Almtx
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

PAD = 0

def get_len_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    seq = seq.ne(PAD)
    seq[:,0] = 1
    return seq.type(torch.float)

def get_attn_key_pad_mask_K(seq_k, seq_q, transpose=False):
    """ For masking out the padding part of key sequence. """
    # [B,L_q,K]
    if transpose:
        seq_q = rearrange(seq_q, 'b l k -> b k l 1')
        seq_k = rearrange(seq_k, 'b l k -> b k 1 l')
    else:
        seq_q = rearrange(seq_q, 'b k l -> b k l 1')
        seq_k = rearrange(seq_k, 'b k l -> b k 1 l')

    return torch.matmul(seq_q, seq_k).eq(PAD)

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s, type_num = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    
    subsequent_mask = rearrange(subsequent_mask, 'l l -> b k l l', b=sz_b, k=type_num)
    return subsequent_mask

class Model(nn.Module):
    '''
    - paper: "Warpformer: A Multi-scale Modeling Approach for Irregular Clinical Time Series" (KDD 2023)
    - paper link: https://dl.acm.org/doi/abs/10.1145/3580305.3599543
    - code adapted from: https://github.com/imJiawen/Warpformer
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
    ):
        
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.embed_time = configs.d_model
        self.pred_len = configs.pred_len_max_irr or configs.pred_len

        num_types = configs.enc_in
        d_inner = 64
        n_head = configs.n_heads
        d_k = 8
        d_v = 8
        dropout = configs.dropout
        warp_num = '0_0.2_1' # ?
        median_len = configs.seq_len_max_irr or configs.seq_len # equal to seq_len_max_irr if not None, else seq_len
        if warp_num is not None:
            warp_num = [float(i) for i in warp_num.split("_")]
            
            if warp_num[-1] > 3:
                warp_num = [int(i) for i in warp_num]
            else:
                warp_num = [int(i * median_len) for i in warp_num]
        else:
            warp_num = []
        self.remove_rep = None
        self.input_only = False
        self.dec_only = False
        self.tau = configs.scale_factor

        # event type embedding
        self.event_enc = Event_Encoder(configs.d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)])
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        self.num_types = num_types
        # self.task = configs.task
        task = configs.dataset_name
        self.task = task
        if len(warp_num) > 0:
            warp_num = warp_num
        elif task == "mor" or task == "wbm":
            warp_num = [0,12]
        else:
            warp_num = [0,6]

        if sum(warp_num) == 0:
            self.no_warping = True
        else:
            self.no_warping = False
        self.full_attn = False

        hourly = False
        if not hourly and len(warp_num) == 0:
            warp_layer_num = 1
        elif hourly:
            warp_layer_num = 2
        else:
            warp_layer_num = len(warp_num)
            
        print("warp_num: ", str(warp_num), "\t warp_layer_num:", str(warp_layer_num))

        self.warpformer_layer_stack = nn.ModuleList([
            Warpformer(int(warp_num[i]), n_head, d_k, d_v, dropout, configs)
            for i in range(warp_layer_num)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=configs.d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)
        
        self.w_t = nn.Linear(1, num_types, bias=False)

        self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)

        self.agg_attention = Attention_Aggregator(configs.d_model, task=task)
        self.agg_attention_wo_feature = Attention_Aggregator_wo_Feature(configs.d_model, task=task)
        self.linear = nn.Linear(configs.d_model*warp_layer_num, configs.d_model)
        if configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            self.linear_predict = nn.Linear(configs.d_model, configs.pred_len_max_irr or configs.pred_len)
        elif configs.task_name == "classification":
            self.decoder_classification = nn.Linear(
                num_types * configs.d_model,
                configs.n_classes
            )
        else:
            raise NotImplementedError

    def forward(
        self, 
        x: Tensor, 
        x_mark: Tensor = None, 
        x_mask: Tensor = None, 
        y: Tensor = None, 
        y_mask: Tensor = None,
        y_class: Tensor = None,
        **kwargs
    ):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''
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
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)
        if y_class is None:
            if self.configs.task_name == "classification":
                logger.warning(f"y_class is missing for the model input. This is only reasonable when the model is testing flops!")
            y_class = torch.ones((BATCH_SIZE), dtype=x.dtype, device=x.device)

        event_time = x_mark[:, :, 0]
        event_value = x
        non_pad_mask = x_mask
        tau = self.cal_tau(event_time, non_pad_mask[:, :, 0])
        # END adaptor

        # embedding
        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d') # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d') # [B,K,L,D]
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        # event_emb = self.type_matrix * non_pad_mask
        event_emb = self.type_matrix
        event_emb = self.event_enc(event_emb) 
        event_emb = rearrange(event_emb, 'b l k d -> b k l d') # [B,K,L,D]

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')
        
        if self.remove_rep == 'abs':
            h0 = value_emb + tau_emb + event_emb
        elif self.remove_rep == 'type':
            h0 = value_emb + tau_emb + tem_enc_k
        elif self.remove_rep == 'rel':
            h0 = value_emb + event_emb + tem_enc_k
        elif self.remove_rep == 'tem':
            h0 = value_emb + event_emb
        else:
            h0 = value_emb + tau_emb + event_emb + tem_enc_k

        amlt_list = []

        if self.input_only:
            # b, k, l, d = h0.shape
            # z0 = output = self.agg_attention(h0, rearrange(non_pad_mask, 'b l k -> b k l 1'))
            # z0 = rearrange(h0, 'b k l d -> (b l) d k')
            # z0 = F.max_pool1d(z0, k).squeeze()
            # z0 = rearrange(z0, '(b l) d -> b d l', b=b, l=l)
            # z0 = F.max_pool1d(z0, l).squeeze(-1)
            z0 = torch.mean(h0, dim=1)
            if self.task != 'HumanActivity':
                z0 = torch.mean(z0, dim=1)
        elif self.dec_only:
            z0 = self.agg_attention(h0, rearrange(non_pad_mask, 'b l k -> b k l 1'))
        else:
            non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l') # [B,K,L]
            
            # if self.task != "HumanActivity":
            #     z0 = self.agg_attention(h0, rearrange(non_pad_mask, 'b k l -> b k l 1')) # [B D]
            # else:
            #     z0 = h0 # [B K L D]
            z0 = None
            # warpformer, the first layer is identical warping
            idwarp=True
            for i, enc_layer in enumerate(self.warpformer_layer_stack):
                if enc_layer.new_l == 0:
                    idwarp=True
                else:
                    idwarp=False
                if i > 0 and self.no_warping and self.full_attn:
                    non_pad_mask = torch.ones_like(non_pad_mask).to(non_pad_mask.device)

                h0, non_pad_mask, almat = enc_layer(h0, non_pad_mask, event_time, id_warp=idwarp)

                output = self.agg_attention(h0, rearrange(non_pad_mask, 'b k l -> b k l 1')) 

                if z0 is not None and z0.shape == output.shape:
                    z0 = z0 + output
                else:
                    z0 = output

                if almat is not None:
                    amlt_list.append(almat.detach().cpu())
            
        output = self.agg_attention_wo_feature(h0, rearrange(non_pad_mask, 'b k l -> b k l 1')) # (B, K, d_model)
        if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
            output = rearrange(self.linear_predict(output), "B K L -> B L K")
            f_dim = -1 if self.configs.features == 'MS' else 0
            PRED_LEN = y.shape[1]
            return {
                "pred": output[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }
        elif self.configs.task_name == "classification":
            return {
                "pred_class": self.decoder_classification(output.view(BATCH_SIZE, -1)),
                "true_class": y_class
            }
        else:
            raise NotImplementedError

    def cal_tau(self, observed_tp, observed_mask):
        '''
        calculate time differences among observations
        '''
        # input: observed_tp [B, L, K], observed_mask [B, L]
        if observed_tp.ndim == 2:
            tmp_time = observed_mask.unsqueeze(-1) * observed_tp.unsqueeze(-1)  # [B, L, K]
        else:
            tmp_time = observed_tp.clone()

        b, l, k = tmp_time.shape

        new_mask = observed_mask.clone()
        new_mask[:, 0] = 1
        tmp_time[new_mask == 0] = float('nan')  # Set masked values to NaN

        # Padding missing values with the next valid value
        for i in range(l):
            if i > 0:
                nan_mask = torch.isnan(tmp_time[:, i])
                tmp_time[nan_mask, i] = tmp_time[nan_mask, i - 1]

        # Calculate differences
        tmp_time_diff = tmp_time[:, 1:] - tmp_time[:, :-1]  # [B, L-1, K]
        
        # Adjust the shape to match original dimensions
        tmp_time_diff = torch.cat((torch.zeros(b, 1, k, device=tmp_time_diff.device), tmp_time_diff), dim=1)  # [B, L, K]

        return tmp_time_diff * observed_mask.unsqueeze(-1)  # Apply the mask



class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(FFNN, self).__init__()

        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x

class Value_Encoder(nn.Module):
    def __init__(self, hid_units, output_dim, num_type):
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.num_type = num_type
        super(Value_Encoder, self).__init__()

        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask

class Event_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Event_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=PAD)

    def forward(self, event):
        # event = event * self.type_matrix
        event_emb = self.event_emb(event.long())
        return event_emb

class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else: # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1) # [B,L,1,D]
        out = torch.mul(out, self.k_map)
        # return out * non_pad_mask # [B,L,K,D]
        return out

class MLP_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, num_types, hid_dim=16):
        super(MLP_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else: # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        # out1 = F.gelu(self.linear1(tt))
        tt = self.encoder(tt)
        tt = torch.mul(tt, self.k_map)
        return tt * non_pad_mask # [B,L,K,D]

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        
        super().__init__()
        self.opt = opt
        self.d_model = d_model
        self.embed_time = d_model

        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)]).to(opt.device)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        
        self.num_types = num_types

        # self.enc_tau = opt.enc_tau
        self.a = nn.Parameter(torch.ones(1,num_types,1,1))
        self.b = nn.Parameter(torch.ones(1,num_types,1,1))
        self.sigma = nn.Parameter(torch.ones(1,num_types,1,1))
        self.pi = torch.tensor(math.pi)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, configs=opt)
            for _ in range(n_layers)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)

        self.w_t = nn.Linear(1, num_types, bias=False)
            
        self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)
            
        self.agg_attention = Attention_Aggregator(d_model, task=opt.task)


    def forward(self, event_time, event_value, non_pad_mask, tau=None, diffseg=None):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''

        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d') # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d') # [B,K,L,D]
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        event_emb = self.type_matrix * non_pad_mask
        event_emb = self.event_enc(event_emb) 
        event_emb = rearrange(event_emb, 'b l k d -> b k l d') # [B,K,L,D]

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')
        k_output = value_emb + tau_emb + event_emb + tem_enc_k
        
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l') # [B,K,L]

        for enc_layer in self.layer_stack:
            k_output, _, _ = enc_layer(
                k_output,
                non_pad_mask=non_pad_mask) 

        non_pad_mask = rearrange(non_pad_mask, 'b k l -> b k l 1') # [B,K,L]
        output = self.agg_attention(k_output, non_pad_mask) # [B,D]
        # k_output = rearrange(k_output, 'b k l d -> b l k d')
        return output

class Warpformer(nn.Module):
    def __init__(self, new_l, n_head, d_k, d_v, dropout, configs: ExpConfigs):
        super().__init__()
        self.new_l = new_l
        self.num_types = 32
        self.configs = configs
        d_inner = 64
        d_model = configs.d_model
        
        self.hourly = False
        if not self.hourly:
            self.get_almtx = Almtx(configs, self.new_l)

        else:
            self.time_split = [i for i in range(self.new_l+1)] # for hour aggregation

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, configs=configs)
            for _ in range(configs.n_layers)])

    
    def hour_aggregate(self, event_time, h0, non_pad_mask):
        # time_split: [new_l]
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]
        new_l = len(self.time_split)-1
        b, k, l, dim = h0.shape
        
        event_time_k = repeat(event_time, 'b l -> b k l', k=self.num_types)
        new_event_time = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        new_h0 = torch.zeros((b,self.num_types,new_l, dim)).to(h0.device)
        new_pad_mask = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        almat = torch.zeros((b,l,new_l)).to(h0.device) # [B,L,S]
        
        # for each time slot
        for i in range(len(self.time_split)-1):
            idx = (event_time_k.ge(self.time_split[i]) & event_time_k.lt(self.time_split[i+1])) # [B,K,L]
            total = torch.sum(idx, dim=-1) # [B,K]
            total[total==0] = 1
            
            tmp_h0 = h0 * idx.unsqueeze(-1) # [B,K,L,D]
            tmp_h0 = rearrange(tmp_h0, 'b k l d -> (b k) d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze() # [BK,D,1]
            new_h0[:,:,i,:] = rearrange(tmp_h0, '(b k) d -> b k d', b=b)
            almat[:,:,i] = (event_time.ge(self.time_split[i]) & event_time.lt(self.time_split[i+1])) # [B,L]

            new_event_time[:,:,i] = torch.sum(event_time_k * idx, dim=-1) / total
            new_pad_mask[:,:,i] = torch.sum(non_pad_mask * idx, dim=-1) / total
        
        almat = repeat(almat, 'b l s -> b k l s', k=k)
        return new_h0, new_event_time, new_pad_mask, almat
    
    def almat_aggregate(self, event_time, h0, non_pad_mask):
        # K: the number of clusters
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]

        b, k, l, dim = h0.shape
        new_event_time = None

        bound_mask, almat = self.get_almtx(h0, mask=non_pad_mask) # [bk, s, l], [bk, s, l]
            
        # almat = almat.to(h0.device)

        almat = rearrange(almat, '(b k) s l -> b k s l', k=k)
        bound_mask = rearrange(bound_mask, '(b k) s l -> b k s l', k=k)
        new_h0 = torch.matmul(almat, h0) # [b k s d]
        
        # # calculate new_event_time
        # event_time = repeat(event_time, 'b l -> b k l', k=(k*self.new_l))
        # event_time = repeat(event_time, 'b (k s) l -> b k l s', k=k, s=self.new_l)
        # new_event_time = event_time * hard_amlt
        # new_event_time = torch.sum(new_event_time, axis=-2)
        # new_event_time = new_event_time / torch.sum(hard_amlt, axis=-2)
        # new_event_time = torch.nan_to_num(new_event_time)
        
        # non_pad_mask = repeat(non_pad_mask, 'b k l -> b k l s', s=self.new_l)
        new_pad_mask = torch.sum(bound_mask, dim=-1)
        new_pad_mask[new_pad_mask > 0] = 1
        new_pad_mask = torch.nan_to_num(new_pad_mask)

        return new_h0, new_event_time, new_pad_mask, almat
    
    def forward(self, h0, non_pad_mask, event_time, id_warp=False):
        if id_warp:
            z0 = h0
            new_pad_mask = non_pad_mask
            almat = None
        else:
            # get alignment matrix
            if self.hourly:
                z0, _, new_pad_mask, almat = self.hour_aggregate(event_time, h0, non_pad_mask)
            else:
                z0, _, new_pad_mask, almat = self.almat_aggregate(event_time, h0, non_pad_mask)
        
        for enc_layer in self.layer_stack:
            z0, _, _ = enc_layer(z0, non_pad_mask=new_pad_mask)
        
        return z0, new_pad_mask, almat

class Pool_Classifier(nn.Module):

    def __init__(self, dim, cls_dim):
        super(Pool_Classifier, self).__init__()
        self.classifier = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D]
        """
        b, l, k = ENCoutput.size(0), ENCoutput.size(1), ENCoutput.size(2)
        ENCoutput = rearrange(ENCoutput, 'b l k d -> (b l) d k')
        ENCoutput = F.max_pool1d(ENCoutput, k).squeeze()
        ENCoutput = rearrange(ENCoutput, '(b l) d -> b d l', b=b, l=l)
        ENCoutput = F.max_pool1d(ENCoutput, l).squeeze(-1)
        return self.classifier(ENCoutput)
    
class Attention_Aggregator(nn.Module):
    def __init__(self, dim, task):
        super(Attention_Aggregator, self).__init__()
        self.task = task
        self.attention_len = Attention(dim*2, dim)
        self.attention_type = Attention(dim*2, dim)

    def forward(self, ENCoutput, mask):
        """
        input: [B,K,L,D], mask: [B,K,L]
        """
        if self.task == "HumanActivity":
            mask = rearrange(mask, 'b k l 1 -> b l k 1')
            ENCoutput = rearrange(ENCoutput, 'b k l d -> b l k d')
            ENCoutput, _ = self.attention_type(ENCoutput, mask) # [B L D]
        else:
            ENCoutput, _ = self.attention_len(ENCoutput, mask) # [B,K,D]
            ENCoutput, _ = self.attention_type(ENCoutput) # [B,D]
        return ENCoutput

class Attention_Aggregator_wo_Feature(nn.Module):
    def __init__(self, dim, task):
        super(Attention_Aggregator_wo_Feature, self).__init__()
        self.task = task
        self.attention_len = Attention(dim*2, dim)

    def forward(self, ENCoutput, mask):
        """
        input: [B,K,L,D], mask: [B,K,L]
        """
        ENCoutput, _ = self.attention_len(ENCoutput, mask) # [B,K,D]
        return ENCoutput
    
class Classifier(nn.Module):

    def __init__(self, dim, type_num, cls_dim, activate=None):
        super(Classifier, self).__init__()
        # self.linear1 = nn.Linear(dim, type_num)
        # # self.activate = nn.Sigmoid()
        # self.linear2 = nn.Linear(type_num, cls_dim)
        self.linear = nn.Linear(dim, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D], mask: [B,L,K]
        """
        # ENCoutput = self.linear1(ENCoutput)
        # # if self.activate:
        # #     ENCoutput = self.activate(ENCoutput)
        # ENCoutput = self.linear2(ENCoutput)
        ENCoutput = self.linear(ENCoutput)
        return ENCoutput

