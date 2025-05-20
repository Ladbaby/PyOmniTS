import math
import string
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class TimeBERT(nn.Module):
 
    def __init__(self):
        super(TimeBERT, self).__init__()

        self.periodic = nn.Linear(1, 63)
        self.linear = nn.Linear(1, 1)


    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


    def forward(self, time_steps):

        key_learn = self.learn_time_embedding(time_steps)
        print(key_learn.shape)


'''
time = torch.rand(9)
model = TimeBERT()
print(time.shape)

model(time)
'''


class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.h = num_heads
        self.embed_time_k = embed_time
        self.linears = nn.ModuleList([nn.Linear(2, 7), 
                                      nn.Linear(3, 9),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        
    
    def forward(self, query, key):
        print('Query:' + str(query.shape) + ', Key:' + str(key.shape))

        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        
        '''
        for l, x in zip(self.linears, (query, key)):
            print('X:' + str(x.shape))
            y = l(x)
            print('Y:' + str(y.shape))
        '''

        print('Query:' + str(query.shape) + ', Key:' + str(key.shape))


'''
hidden_size = 16
embed_time = 16
num_heads = 1
batch, seq_len, input_dim = 5, 20, 3


query = torch.rand(batch, seq_len, 2)
key = torch.rand(batch, seq_len, 3)

model = multiTimeAttention(2 * input_dim, hidden_size, embed_time, num_heads)
model(query, key)
'''



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


x, y = torch.rand(8, 4), torch.rand(8, 4)
model = Similarity(temp=1)

# input becomes x : (8, 1, 4), y : (1, 8, 4)
cos_sim = model(x.unsqueeze(1), y.unsqueeze(0))
print(cos_sim.shape)

labels = torch.arange(cos_sim.size(0)).long()
print(labels)