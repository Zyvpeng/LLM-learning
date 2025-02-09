import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0

        self.d_k = self.d_model//self.num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,masked_attention=None):
        batchsize = q.size(0)
        qk = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.d_k)
        if masked_attention is not None:
            qk = qk.masked_fill(masked_attention.unsqueeze(1)==0,float('-inf'))
            # print(qk)
        weights = F.softmax(qk,dim=-1)   #(batch_size,num_heads,legnth_q,length_k)
        attention = torch.matmul(weights,v)   #(batch_size,num_heads,legnth_q,d_k)
        return attention

        #q k v (batch_size,seq_length, d_model)
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,masked_attention:torch.Tensor=None):
        batchsize = q.size(0)
        #(batch_size,num_heads.seq_length,d_k)
        q = q.view(batchsize,-1,self.num_heads,self.d_k).transpose(1,2)
        k = k.view(batchsize,-1,self.num_heads,self.d_k).transpose(1,2)
        v = v.view(batchsize,-1,self.num_heads,self.d_k).transpose(1,2)
        attention = self.scaled_dot_product_attention(q,k,v,masked_attention)

        attention = attention.transpose(1,2).contiguous().view(batchsize,-1,self.d_model)
        output = self.W_o(attention)
        return output



q = torch.randn(2,10,512)
k = torch.randn(2,15,512)
v = torch.randn(2,15,512)
masked_attention = torch.ones(2,10,15)
masked_attention = torch.tril(masked_attention)
print(masked_attention)
d_model = 512
num_heads = 8

model = MultiHeadAttention(d_model,num_heads)
output = model(q,k,v,masked_attention)





