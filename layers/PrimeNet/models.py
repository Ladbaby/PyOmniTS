import math
import string
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import TransformerBlock, SwitchTransformerBlock



class TimeBERTConfig:
    def __init__(
        self,
        input_dim:int,
        dataset:string=None,
        pretrain_tasks:string=None,
        cls_query:torch.Tensor=torch.linspace(0, 1., 128),
        hidden_size:int=16,
        embed_time:int=16,
        num_heads:int=1,
        learn_emb:bool=True,
        freq:float=10.,
        pooling:str='ave',
        classify_pertp:bool=False,
        max_length:int=128,
        dropout:float=0.3,
        temp:float=0.05,
        switch_keys:List=['pretraining', 'classification'],
        n_classes: int = 2
    ):

        self.dataset=dataset
        self.pretrain_tasks=pretrain_tasks
        self.input_dim=input_dim
        self.cls_query=cls_query
        self.hidden_size=hidden_size
        self.embed_time=embed_time
        self.num_heads=num_heads
        self.learn_emb=learn_emb
        self.freq=freq
        self.pooling=pooling
        self.classify_pertp=classify_pertp,
        self.max_length=max_length
        self.dropout=dropout
        self.temp=temp
        self.switch_keys=switch_keys
        self.n_classes=n_classes



class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), # to embed the query
                                      nn.Linear(embed_time, embed_time), # to embed the key
                                      nn.Linear(input_dim*num_heads, nhidden)]) # to embed attention weighted values
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        '''
        key: (176, 8, 3, 16)
        query: (1, 8, 128, 16)
        value: (176, 1, 3, 16)
        mask: (176, 1, 3, 0)
        '''
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.to(query.device).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn.to(query.device)*value.unsqueeze(-3).to(query.device), -2), p_attn.to(query.device)
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        # zip will pair the 3 linear layers with (query, key)
        # Since there are 3 linear layers but only 2 elements in the tupe (query, key),
        # only the first two linear layers will be mapped to this tuple
        # (first linear layer, query), (second linear layer, key)
        # so the list has two elements
        # input query passed through the first linear layer -> output becomes first element of the list (query)
        # input key passed through the second linear layer -> output becomes second element of the list (key)
        # query, key = [2, 3]
        # so 2 is assigned to query and 3 is assigned to key
        # had there been a, b, c = [1, 2] -> ValueError: not enough values to unpack (expected 3, got 2)
        # had there been a, b = [1, 2, 3] -> ValueError: too many values to unpack (expected 2)
        # look into learn_time_emb.py for further clarification on this line
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)



class BertPooler(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, first_token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertInterpHead(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * config.hidden_size, config.input_dim)

    def forward(self, first_token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


'''
sim = Similarity(temp=0.05)
z1, z2 = torch.rand(5, 1, 3), torch.rand(1, 5, 3)
cos_sim = sim(z1, z2)
print(z1)
print(z2)
print(cos_sim)
'''


class TimeBERT(nn.Module):
 
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERT, self).__init__()
       
        assert config.embed_time % config.num_heads == 0
        self.config = config
        self.freq = config.freq
        self.embed_time = config.embed_time
        self.learn_emb = config.learn_emb
        self.dim = config.input_dim
        self.hidden_size = config.hidden_size
        self.cls_query =  config.cls_query
        self.time_att = multiTimeAttention(2*self.dim, self.hidden_size, self.embed_time, config.num_heads)

        self.pos_emb = nn.Embedding(config.max_length, self.hidden_size)  ## +1 for cls token
        self.transformer = TransformerBlock(self.hidden_size, config.num_heads, self.hidden_size, dropout=config.dropout)

        self.cls_emb = nn.Embedding(1, self.hidden_size)
        self.pooling = config.pooling

        self.pooler = BertPooler(config)

        assert self.pooling in ['ave', 'att', 'bert']

        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    
    def encode(self, x, is_pooling=False):
        out = x

        if self.pooling == 'att' or self.pooling == 'bert':

            batch_size = out.size(0)
            # cls_tokens = torch.zeros(batch_size).to(out.device)
            cls_tokens = torch.zeros(batch_size).to(out.device).long()
            cls_repr = self.cls_emb(cls_tokens).view(batch_size, 1, -1)  # (batch_size, 1, nhidden)

        if self.pooling == 'bert':
            out = torch.cat([cls_repr, out], dim=1)

        if is_pooling:
            out = self.transformer(out)
            if self.pooling == 'ave':
                out = torch.mean(out, dim=1)  #Ave Pooling
            elif self.pooling == 'att':
                out = out.permute(0, 2, 1) # (batch_size, seq_len, nhidden) -> (batch_size, nhidden, seq_len)
                weights = F.softmax(torch.bmm(cls_repr, out), dim=-1) # (batch_size, 1, seq_len)
                out = torch.sum(out * weights, dim=-1) # (batch_size, nhidden)
            else: # bert
                out = out[ : , 0]

            return self.pooler(out)

        else:
            
            positions = torch.arange(out.shape[1]).long().unsqueeze(0).repeat([out.shape[0], 1])
            out = out + self.pos_emb(positions.to(out.device))
            out = self.transformer(out)
            if self.pooling == 'bert':
                # return out[1:]  ## remove cls token
                return out[:, 1:]
            else:
                return out
    
       
    def forward(self, x, time_steps, query_time_steps=None):

        '''
        x: (176, 3, 36)
        time_steps: (176, 3)
        '''
        
        # x : (batch x num_seq), seq_len, (input_dim x 2)
        # time_steps : (batch x num_seq), seq_len
        # query_time_steps : (batch x num_seq), seq_len, input_dim

        device = x.device
        x, time_steps = x.float(), time_steps.float()
              
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)

        if query_time_steps is None:
            query_time_steps = time_steps
        
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            time_query = self.learn_time_embedding(query_time_steps)
            cls_query = self.learn_time_embedding(self.cls_query.unsqueeze(0).to(device))
        else:
            key = self.time_embedding(time_steps.cpu(), self.embed_time).to(device)
            time_query = self.time_embedding(query_time_steps.cpu(), self.embed_time).to(device)
            cls_query = self.time_embedding(self.cls_query.unsqueeze(0), self.embed_time).to(device)
        
        # time_query maintains the original timestamps in the data.
        # This is useful for interpolation or per timestamp classification because interpolation or the classification will be done at the irregular times themselves

        # cls_query transforms the irregular into a regular time series by putting the query timestamps uniformly.
        # This will be useful for tasks where the overall representation is important for the end task and not the per timestamp representation.
        # So if we use the cls_query, although the time-series is irregular, the learnt representation from cls_query will that be of a corresponding regular time-series.

        # time-attention computes attention between timestamps and feature values 

        # key comes from the timestamps of the data
        # query comes from the fixed vector torch.linspace(0, 1., 128)
        # In other words, I'm querying the original timestamps of the data, which is the key, to build the time representation of uniformly spaced time between [0, 0.006, ... 1]
        # cls_out -> uses cls_query to transform the irregular time-series into the corresponding regular time-series representation

        # time_query -> irregular time representation
        # cls_query -> corresponding regular time representation
        cls_out = self.time_att(cls_query, key, x, mask) 

        # both query and key comes from the timestamps of the data
        # out -> uses time_query to transform the irregular time-series into the corresponding irregular time-series representation
        out = self.time_att(time_query, key, x, mask) 


        # self-attention computes attention between feature values and feature values, that's why self.
        
        # since cls_out comes from cls_query, the learnt feature representation is for the corresponding regular time-series
        # cls_out is pooled ('ave', 'att', 'bert') and the output is used for CL representation and sequence classification
        cls_out = self.encode(cls_out, is_pooling=True)
        
        # since out comes time_query, the learnt feature representation is for the corresponding irregular time-series
        # unpooled output is used for Interpolation representation and per timestep classification
        out = self.encode(out, is_pooling=False)

        return {'cls_pooling': cls_out, 'last_hidden_state': out}



class SwitchTimeBERT(nn.Module):
 
    def __init__(self, config: TimeBERTConfig):
        super(SwitchTimeBERT, self).__init__()
       
        assert config.embed_time % config.num_heads == 0
        self.config = config
        self.freq = config.freq
        self.embed_time = config.embed_time
        self.learn_emb = config.learn_emb
        self.dim = config.input_dim
        self.hidden_size = config.hidden_size
        self.cls_query =  config.cls_query
        self.time_att = multiTimeAttention(2*self.dim, self.hidden_size, self.embed_time, config.num_heads)

        self.pos_emb = nn.Embedding(config.max_length, self.hidden_size)  ## +1 for cls token
        switch_keys = [key+'_pooling' for key in config.switch_keys] + config.switch_keys # updated line for switch transformer
        self.transformer = SwitchTransformerBlock(self.hidden_size, config.num_heads, self.hidden_size, dropout=config.dropout, switch_keys=switch_keys)

        self.cls_emb = nn.Embedding(1, self.hidden_size)
        self.pooling = config.pooling

        self.pooler = BertPooler(config)

        assert self.pooling in ['ave', 'att', 'bert']

        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    
    def encode(self, x, switch_key, is_pooling=False):

        out = x

        if self.pooling == 'att' or self.pooling == 'bert':

            batch_size = out.size(0)
            # cls_tokens = torch.zeros(batch_size).to(out.device)
            cls_tokens = torch.zeros(batch_size).to(out.device).long()
            cls_repr = self.cls_emb(cls_tokens).view(batch_size, 1, -1)  # (batch_size, 1, nhidden)

        if self.pooling == 'bert':
            out = torch.cat([cls_repr, out], dim=1)

        if is_pooling:
            out = self.transformer(out, switch_key=switch_key+'_pooling')
            if self.pooling == 'ave':
                out = torch.mean(out, dim=1)  #Ave Pooling
            elif self.pooling == 'att':
                out = out.permute(0, 2, 1) # (batch_size, seq_len, nhidden) -> (batch_size, nhidden, seq_len)
                weights = F.softmax(torch.bmm(cls_repr, out), dim=-1) # (batch_size, 1, seq_len)
                out = torch.sum(out * weights, dim=-1) # (batch_size, nhidden)
            else: # bert
                out = out[ : , 0]

            return self.pooler(out)

        else:
            
            positions = torch.arange(out.shape[1]).long().unsqueeze(0).repeat([out.shape[0], 1])
            out = out + self.pos_emb(positions.to(out.device))
            out = self.transformer(out, switch_key=switch_key)
            if self.pooling == 'bert':
                # return out[1:]  ## remove cls token
                return out[:, 1:]

            else:
                return out
    
       
    def forward(self, x, time_steps, switch_key, query_time_steps=None):
        
        # x : (batch x num_seq), seq_len, (input_dim x 2)
        # time_steps : (batch x num_seq), seq_len
        # query_time_steps : (batch x num_seq), seq_len, input_dim

        device = x.device
        x, time_steps = x.float(), time_steps.float()
              
        mask = x[:, :, self.dim:]
        mask = torch.cat((mask, mask), 2)

        if query_time_steps is None:
            query_time_steps = time_steps
        
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            time_query = self.learn_time_embedding(query_time_steps)
            cls_query = self.learn_time_embedding(self.cls_query.unsqueeze(0).to(device))
        else:
            key = self.time_embedding(time_steps.cpu(), self.embed_time).to(device)
            time_query = self.time_embedding(query_time_steps.cpu(), self.embed_time).to(device)
            cls_query = self.time_embedding(self.cls_query.unsqueeze(0), self.embed_time).to(device)
        
        cls_out = self.time_att(cls_query, key, x, mask)
        out = self.time_att(time_query, key, x, mask)

        cls_out = self.encode(cls_out, switch_key=switch_key, is_pooling=True) # pooled ('ave', 'att', 'bert') output is used for CL representation and sequence classification
        out = self.encode(out, switch_key=switch_key, is_pooling=False) # unpooled output is used for Interpolation representation and per timestep classification

        return {'cls_pooling': cls_out, 'last_hidden_state': out}

def isnan(x):
    return torch.any(torch.isnan(x))

class TimeBERTForPretraining(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForPretraining, self).__init__()

        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)
        self.interp_head = BertInterpHead(config)
        self.sim = Similarity(temp=config.temp)

    def forward(self, x, time_steps):
        '''
        x: batch_size, num_seq, seq_len, (input_dim x 2)
        time_steps: (batch_size, num_seq, seq_len)
        '''
        # print(x.shape)
        batch_size = x.size(0)
        num_seq = x.size(1)
        seq_length = x.size(2)
        input_size = x.size(3)

        mask = x[:, :, :, self.dim:].float()

        query_time_steps = torch.cat([time_steps[:, 1].unsqueeze(1), time_steps[:, 0].unsqueeze(1)], dim=1)
        # print(query_time_steps.shape) # batch, num_seq, seq_len
        
        interp_labels = torch.cat([x[:, 1, :, :self.dim].unsqueeze(1), x[:, 0, :, :self.dim].unsqueeze(1)], dim=1).view((batch_size * num_seq, seq_length, self.dim))
        # print(interp_labels.shape) # (batch x num_seq), seq_len, input_dim

        interp_mask = torch.cat([mask[:, 1, :, :].unsqueeze(1), mask[:, 0, :, :].unsqueeze(1)], dim=1).view((batch_size * num_seq, seq_length, self.dim))
        # print(interp_mask.shape) # (batch x num_seq), seq_len, input_dim

        x = x.view((batch_size * num_seq, seq_length, input_size))
        # print(x.shape) # (batch x num_seq), seq_len, (input_dim x 2)

        time_steps = time_steps.view((batch_size * num_seq, seq_length))
        # print(time_steps.shape) # (batch x num_seq), seq_len

        query_time_steps = query_time_steps.view((batch_size * num_seq, seq_length))
        # print(query_time_steps.shape) # (batch x num_seq), seq_len, input_dim

        # x : (batch x num_seq), seq_len, (input_dim x 2) -> (v1, m1) followed by (v2, m2)
        # time_steps : (batch x num_seq), seq_len
        # query_time_steps : (batch x num_seq), seq_len, input_dim
        outputs = self.bert(x, time_steps, query_time_steps)

        cls_pooling = outputs['cls_pooling'] 
        # print(cls_pooling.shape) # (batch_size x num_seq), hidden_size
        
        last_hidden_state = outputs['last_hidden_state'] 
        # print(last_hidden_state.shape) # (batch_size x num_seq), seq_len, hidden_size

        interp_output = self.interp_head(last_hidden_state) 
        # print(interp_output.shape) # (batch_size x num_seq), seq_len, input_dim

        cls_pooling = cls_pooling.view((batch_size, num_seq, self.config.hidden_size))
        # print(cls_pooling.shape) # (batch_size, num_seq, hidden_size)

        # contrastive learning
        z1, z2 = cls_pooling[:, 0], cls_pooling[:, 1]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # print(cos_sim) # batch_size, batch_size

        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device) 
        # print(labels) # batch_size

        loss_fct = nn.CrossEntropyLoss()
        cl_loss = loss_fct(cos_sim, labels)

        correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()

        # interpolation
        # interp_output : (batch x num_seq), seq_len, input_dim -> (output of v1) followed by (output of v2)
        # interp_labels : (batch x num_seq), seq_len, input_dim -> v2 followed by v1
        # interp_mask : (batch x num_seq), seq_len, input_dim -> m2 followed by m1
        # both subsamples are used to interpolate one from the other
        mse_loss = torch.sum(((interp_output-interp_labels)*interp_mask)**2.0)  / torch.sum(interp_mask)

        if self.config.pretrain_tasks == 'cl':
            loss = cl_loss
        elif self.config.pretrain_tasks == 'interp':
            loss = mse_loss
        else:
            loss = cl_loss + mse_loss

        return {'loss': loss, 'cl_loss': cl_loss, 'mse_loss': mse_loss, 'correct_num': correct_num, 'total_num': batch_size}


class TimeBERTForPretrainingV2(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForPretrainingV2, self).__init__()

        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)
        self.interp_head = BertInterpHead(config)
        self.sim = Similarity(temp=config.temp)

    def forward(self, x, time_steps):
        '''
        x : batch_size, num_seq, seq_len, (input_dim x 3)
        time_steps : batch_size, num_seq, seq_len
        '''
        #print('0 ', isnan(x),isnan(time_steps))
        x, interp_mask = x[:, :, :, :-self.dim], x[:, :, :, -self.dim:].float()
        # x : batch_size, num_seq, seq_len, (input_dim x 2)
        # interp_mask : batch_size, num_seq, seq_len, input_dim

        batch_size = x.size(0)
        num_seq = x.size(1)
        seq_length = x.size(2)
        input_size = x.size(3)

        interp_labels = x[:, :, :, :self.dim].view((batch_size * num_seq, seq_length, self.dim))
        # interp_labels : (batch_size x num_seq), seq_len, input_dim
        
        x = x.view((batch_size * num_seq, seq_length, input_size))
        # x : (batch_size x num_seq), seq_len, (input_dim x 2)

        time_steps = time_steps.view((batch_size * num_seq, seq_length))
        # time_steps : (batch_size x num_seq), seq_len

        interp_mask = interp_mask.view((batch_size * num_seq, seq_length, self.dim))
        # interp_mask : (batch_size x num_seq), seq_len, input_dim
        
        outputs = self.bert(x, time_steps)
        
        cls_pooling = outputs['cls_pooling'] 
        # cls_pooling : (batch_size x num_seq), hidden_size
        
        last_hidden_state = outputs['last_hidden_state'] 
        # last_hidden_state : (batch_size x num_seq), seq_len, hidden_size

        #print('1 ', isnan(cls_pooling),isnan(last_hidden_state))

        interp_output = self.interp_head(last_hidden_state)
        # interp_output : (batch_size x num_seq), seq_len, input_dim

        cls_pooling = cls_pooling.view((batch_size, num_seq, self.config.hidden_size))
        # cls_pooling.shape : batch_size, num_seq, hidden_size

        # contrastive learning
        z1, z2 = cls_pooling[:, 0], cls_pooling[:, 1]
        # z1, z2 : (batch_size, hidden_size), (batch_size, hidden_size)
        # After unsqueezing -> z1, z2 : (batch_size, 1, hidden_size), (1, batch_size, hidden_size)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # cos_sim : batch_size, batch_size

        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        # labels : batch_size, [0, 1, 2, ... , batch_size]

        loss_fct = nn.CrossEntropyLoss()
        cl_loss = loss_fct(cos_sim, labels)

        correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()

        # interpolation
        # interp_output : (batch x num_seq), seq_len, input_dim -> (output of v1) followed by (output of v2)
        # interp_labels : (batch x num_seq), seq_len, input_dim -> v1 followed by v2
        # interp_mask : (batch x num_seq), seq_len, input_dim -> (interp_mask of v1) followed by (interp_mask of v2)
        mse_loss = torch.sum(((interp_output-interp_labels)*interp_mask)**2.0)  / (torch.sum(interp_mask)+1e-10)

        #print('2 ', isnan(cl_loss),isnan(mse_loss))

        if self.config.pretrain_tasks == 'cl': 
            loss = cl_loss
        elif self.config.pretrain_tasks == 'interp':
            loss = mse_loss
        else:
            loss = cl_loss + mse_loss

        #exit(0)

        return {'loss': loss, 'cl_loss': cl_loss, 'mse_loss': mse_loss, 'correct_num': correct_num, 'total_num': batch_size, "pred": interp_output, "cls_pooling": cls_pooling, "last_hidden_state": last_hidden_state}



class TimeBERTForClassification(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForClassification, self).__init__()
        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, config.n_classes)
        )

    def forward(self, x, time_steps):
        # align with the shape in TimeBERTForPretrainingV2 forward()
        x, interp_mask = x[:, :, :, :-self.dim], x[:, :, :, -self.dim:].float()
        batch_size = x.size(0)
        num_seq = x.size(1)
        seq_length = x.size(2)
        input_size = x.size(3)
        x = x.view((batch_size * num_seq, seq_length, input_size))
        time_steps = time_steps.view((batch_size * num_seq, seq_length))

        outputs = self.bert(x, time_steps)

        # if self.config.dataset == 'physionet' or self.config.dataset == 'MIMIC-III' or (self.config.dataset == 'PersonActivity' and not self.config.classify_pertp[0]):
        #     cls_pooling = outputs['cls_pooling'] # batch_size, hidden_size
        # elif self.config.dataset == 'PersonActivity' and self.config.classify_pertp[0]:
        #     cls_pooling = outputs['last_hidden_state'] # batch_size, seq_len, hidden_size
        
        cls_pooling = outputs['cls_pooling']

        # reduce the n_patch dimension
        return self.classifier(cls_pooling.reshape(batch_size, -1, self.config.hidden_size).mean(1))



class TimeBERTForRegression(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForRegression, self).__init__()
        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)

        if self.config.dataset == 'AE':
            self.regressor = nn.Sequential(
                                nn.Linear(config.hidden_size, 512),
                                nn.ReLU(),

                                nn.Linear(512, 1))

        elif self.config.dataset == 'BC':
            self.regressor = nn.Sequential(
                                nn.Linear(config.hidden_size, 32768),
                                nn.ReLU(),

                                nn.Linear(32768, 1))


    def forward(self, x, time_steps):

        outputs = self.bert(x, time_steps)
        cls_pooling = outputs['cls_pooling'] # batch_size, hidden_size
        return self.regressor(cls_pooling)



class TimeBERTForMultiTask(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForMultiTask, self).__init__()
        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)

        # for pretraining
        self.interp_head = BertInterpHead(config)
        self.sim = Similarity(temp=config.temp)

        # for classification
        if self.config.dataset == 'PersonActivity':
            self.classifier = nn.Sequential(
                                nn.Linear(config.hidden_size, 300),
                                nn.ReLU(),
                                nn.Linear(300, 300),
                                nn.ReLU(),
                                nn.Linear(300, 11))
        elif self.config.dataset == 'MIMIC-III' or self.config.dataset == 'physionet':
            self.classifier = nn.Sequential(
                                nn.Linear(config.hidden_size, 300),
                                nn.ReLU(),
                                nn.Linear(300, 300),
                                nn.ReLU(),
                                nn.Linear(300, 2))

    def forward(self, x, time_steps, task='classification'):
        '''
        x : batch_size, num_seq, seq_len, (input_dim x 3)
        time_steps : batch_size, num_seq, seq_len
        '''

        if task == 'pretraining':

            x, interp_mask = x[:, :, :, :-self.dim], x[:, :, :, -self.dim:].float()
            # x : batch_size, num_seq, seq_len, (input_dim x 2)
            # interp_mask : batch_size, num_seq, seq_len, input_dim

            batch_size = x.size(0)
            num_seq = x.size(1)
            seq_length = x.size(2)
            input_size = x.size(3)

            interp_labels = x[:, :, :, :self.dim].view((batch_size * num_seq, seq_length, self.dim))
            # interp_labels : (batch_size x num_seq), seq_len, input_dim
            
            x = x.view((batch_size * num_seq, seq_length, input_size))
            # x : (batch_size x num_seq), seq_len, (input_dim x 2)

            time_steps = time_steps.view((batch_size * num_seq, seq_length))
            # time_steps : (batch_size x num_seq), seq_len

            interp_mask = interp_mask.view((batch_size * num_seq, seq_length, self.dim))
            # interp_mask : (batch_size x num_seq), seq_len, input_dim
            
            outputs = self.bert(x, time_steps)
            
            cls_pooling = outputs['cls_pooling'] 
            # cls_pooling : (batch_size x num_seq), hidden_size
            
            last_hidden_state = outputs['last_hidden_state'] 
            # last_hidden_state : (batch_size x num_seq), seq_len, hidden_size

            interp_output = self.interp_head(last_hidden_state)
            # interp_output : (batch_size x num_seq), seq_len, input_dim

            cls_pooling = cls_pooling.view((batch_size, num_seq, self.config.hidden_size))
            # cls_pooling.shape : batch_size, num_seq, hidden_size

            # contrastive learning
            z1, z2 = cls_pooling[:, 0], cls_pooling[:, 1]
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            # cos_sim : batch_size, batch_size

            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            # labels : batch_size

            loss_fct = nn.CrossEntropyLoss()
            cl_loss = loss_fct(cos_sim, labels)

            correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()

            # interpolation
            # interp_output : (batch x num_seq), seq_len, input_dim -> (output of v1) followed by (output of v2)
            # interp_labels : (batch x num_seq), seq_len, input_dim -> v1 followed by v2
            # interp_mask : (batch x num_seq), seq_len, input_dim -> (interp_mask of v1) followed by (interp_mask of v2)
            mse_loss = torch.sum(((interp_output-interp_labels)*interp_mask)**2.0)  / (torch.sum(interp_mask)+1e-10)

            if self.config.pretrain_tasks == 'cl': 
                loss = cl_loss
            elif self.config.pretrain_tasks == 'interp':
                loss = mse_loss
            else:
                loss = cl_loss + mse_loss

            return {'loss': loss, 'cl_loss': cl_loss, 'mse_loss': mse_loss, 'correct_num': correct_num, 'total_num': batch_size}

        elif task == 'classification':

            outputs = self.bert(x, time_steps)

            if self.config.dataset == 'physionet' or self.config.dataset == 'MIMIC-III' or (self.config.dataset == 'PersonActivity' and not self.config.classify_pertp[0]):
                cls_pooling = outputs['cls_pooling'] # batch_size, hidden_size
            elif self.config.dataset == 'PersonActivity' and self.config.classify_pertp[0]:
                cls_pooling = outputs['last_hidden_state'] # batch_size, seq_len, hidden_size

            return self.classifier(cls_pooling)

        else:

            raise NotImplementedError


class SwitchTimeBERTForMultiTask(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(SwitchTimeBERTForMultiTask, self).__init__()
        self.config = config
        self.dim = config.input_dim
        self.bert = SwitchTimeBERT(config)

        # for pretraining
        self.interp_head = BertInterpHead(config)
        self.sim = Similarity(temp=config.temp)

        # for classification

        if self.config.dataset == 'PersonActivity':
            self.classifier = nn.Sequential(
                                nn.Linear(config.hidden_size, 300),
                                nn.ReLU(),
                                nn.Linear(300, 300),
                                nn.ReLU(),
                                nn.Linear(300, 11))
        elif self.config.dataset == 'MIMIC-III' or self.config.dataset == 'physionet':
            self.classifier = nn.Sequential(
                                nn.Linear(config.hidden_size, 300),
                                nn.ReLU(),
                                nn.Linear(300, 300),
                                nn.ReLU(),
                                nn.Linear(300, 2))

    def forward(self, x, time_steps, task='classification'):
        '''
        x : batch_size, num_seq, seq_len, (input_dim x 3)
        time_steps : batch_size, num_seq, seq_len
        '''

        if task == 'pretraining':

            x, interp_mask = x[:, :, :, :-self.dim], x[:, :, :, -self.dim:].float()
            # x : batch_size, num_seq, seq_len, (input_dim x 2)
            # interp_mask : batch_size, num_seq, seq_len, input_dim

            batch_size = x.size(0)
            num_seq = x.size(1)
            seq_length = x.size(2)
            input_size = x.size(3)

            interp_labels = x[:, :, :, :self.dim].view((batch_size * num_seq, seq_length, self.dim))
            # interp_labels : (batch_size x num_seq), seq_len, input_dim
            
            x = x.view((batch_size * num_seq, seq_length, input_size))
            # x : (batch_size x num_seq), seq_len, (input_dim x 2)

            time_steps = time_steps.view((batch_size * num_seq, seq_length))
            # time_steps : (batch_size x num_seq), seq_len

            interp_mask = interp_mask.view((batch_size * num_seq, seq_length, self.dim))
            # interp_mask : (batch_size x num_seq), seq_len, input_dim
            
            outputs = self.bert(x, time_steps, switch_key=task) # updated line for switch transformer
            
            cls_pooling = outputs['cls_pooling'] 
            # cls_pooling : (batch_size x num_seq), hidden_size
            
            last_hidden_state = outputs['last_hidden_state'] 
            # last_hidden_state : (batch_size x num_seq), seq_len, hidden_size

            interp_output = self.interp_head(last_hidden_state)
            # interp_output : (batch_size x num_seq), seq_len, input_dim

            cls_pooling = cls_pooling.view((batch_size, num_seq, self.config.hidden_size))
            # cls_pooling.shape : batch_size, num_seq, hidden_size

            # contrastive learning
            z1, z2 = cls_pooling[:, 0], cls_pooling[:, 1]
            cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
            # cos_sim : batch_size, batch_size

            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            # labels : batch_size

            loss_fct = nn.CrossEntropyLoss()
            cl_loss = loss_fct(cos_sim, labels)

            correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()

            # interpolation
            # interp_output : (batch x num_seq), seq_len, input_dim -> (output of v1) followed by (output of v2)
            # interp_labels : (batch x num_seq), seq_len, input_dim -> v1 followed by v2
            # interp_mask : (batch x num_seq), seq_len, input_dim -> (interp_mask of v1) followed by (interp_mask of v2)
            mse_loss = torch.sum(((interp_output-interp_labels)*interp_mask)**2.0)  / (torch.sum(interp_mask)+1e-10)

            if self.config.pretrain_tasks == 'cl': 
                loss = cl_loss
            elif self.config.pretrain_tasks == 'interp':
                loss = mse_loss
            else:
                loss = cl_loss + mse_loss

            return {'loss': loss, 'cl_loss': cl_loss, 'mse_loss': mse_loss, 'correct_num': correct_num, 'total_num': batch_size}

        elif task == 'classification':

            outputs = self.bert(x, time_steps, switch_key=task) # updated line for switch transformer

            if self.config.dataset == 'physionet' or self.config.dataset == 'MIMIC-III' or (self.config.dataset == 'PersonActivity' and not self.config.classify_pertp[0]):
                cls_pooling = outputs['cls_pooling'] # batch_size, hidden_size
            elif self.config.dataset == 'PersonActivity' and self.config.classify_pertp[0]:
                cls_pooling = outputs['last_hidden_state'] # batch_size, seq_len, hidden_size

            return self.classifier(cls_pooling)

        else:

            raise NotImplementedError



class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)



class TimeBERTForInterpolation(nn.Module):
    def __init__(self, config: TimeBERTConfig):
        super(TimeBERTForInterpolation, self).__init__()
        self.config = config
        self.dim = config.input_dim
        self.bert = TimeBERT(config)

        if self.config.dataset == 'P12' or self.config.dataset == 'MIMIC_III_DeBrouwer2019':
            self.interpolator = nn.Sequential(nn.Linear(self.config.hidden_size, 4 * self.config.hidden_size),
                                            nn.ReLU(),

                                            nn.Linear(4 * self.config.hidden_size, 4 * self.config.hidden_size),
                                            nn.ReLU(),

                                            nn.Linear(4 * self.config.hidden_size, 4 * self.config.hidden_size),
                                            nn.ReLU(),

                                            nn.Linear(4 * self.config.hidden_size, 4 * self.config.hidden_size),
                                            nn.ReLU(),

                                            nn.Linear(4 * self.config.hidden_size, self.config.input_dim))

        elif self.config.dataset == 'HumanActivity':
            self.interpolator = nn.Sequential(nn.Linear(self.config.hidden_size, 50),
                                              nn.ReLU(),

                                              nn.Linear(50, self.config.input_dim))

        '''
        self.interpolator = nn.Sequential(nn.Linear(self.config.hidden_size, 50), 
                                          nn.ReLU(), 
                                          nn.Linear(50, self.config.input_dim))

    
        self.interpolator = nn.Sequential(nn.Linear(self.config.hidden_size, 4 * self.config.hidden_size),
                                          
                                          Permute(),
                                          nn.BatchNorm1d(4 * self.config.hidden_size),
                                          Permute(),
                                          
                                          nn.Linear(4 * self.config.hidden_size, 4 * self.config.hidden_size),
                                          
                                          Permute(),
                                          nn.BatchNorm1d(4 * self.config.hidden_size),
                                          Permute(),
                                          
                                          nn.Linear(4 * self.config.hidden_size, self.config.input_dim))
        '''


    def forward(self, x, time_steps):

        outputs = self.bert(x, time_steps)
        last_hidden_state = outputs['last_hidden_state'] # batch_size, seq_len, hidden_size (N, L, C)
        return self.interpolator(last_hidden_state)



'''
device = 'cuda:0'
batch, seq_len, input_size = 5, 20, 3

config = TimeBERTConfig(input_dim=input_size,
                        pretrain_tasks='full1',
                        cls_query=torch.linspace(0, 1., 128),
                        hidden_size=16,
                        batch_size=5,
                        embed_time=16,
                        num_heads=1,
                        learn_emb=True,
                        freq=10.0,
                        pooling='ave',
                        max_length=seq_len,
                        dropout=0.3,
                        temp=0.05)

model = TimeBERTForInterpolation(config).to(device)
x = torch.rand(batch, seq_len, input_size)
m = torch.rand(batch, seq_len, input_size)
t = torch.rand(batch, seq_len)
x_batch = torch.cat([x, m], dim=-1)

print(x.shape, t.shape)
o = model(x_batch.to(device), t.to(device))
print(o.shape)
'''

'''
model = TimeBERTForPretrainingV2(config).to(device)

value_batch = torch.rand(batch, 2, seq_len, input_size).to(device)
time_batch = torch.rand(batch, 2, seq_len).to(device)
mask_batch = torch.randint(0, 2, (batch, 2, seq_len, input_size * 2)).to(device) # for TimeBERTForPretrainingV2
# mask_batch = torch.randint(0, 2, (batch, 2, seq_len, input_size)).to(device) # for TimeBERTForPretrainingV1

x_batch = torch.cat([value_batch, mask_batch], dim=-1)
        
print(x_batch.shape)
out = model(x_batch, time_batch)
'''