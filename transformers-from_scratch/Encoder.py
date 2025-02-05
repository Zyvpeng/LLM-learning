from FeedForward import FeedForward
from multihead_attention import MultiHeadAttention
from PositionEncoder import PositionEncoder

import torch.nn as nn
import torch

class EncoderLayer(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.attn = MultiHeadAttention(d_model,heads)
        self.ff = FeedForward(d_model=d_model,dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output = self.attn(x,x,x,mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout=0.1):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size,d_model)
        self.pe = PositionEncoder(d_model,100)
        self.layers = nn.ModuleList([EncoderLayer(d_model,heads,dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,src,mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,mask)
        return self.norm(x)

