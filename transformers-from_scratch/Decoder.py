from FeedForward import FeedForward
from multihead_attention import MultiHeadAttention
from PositionEncoder import PositionEncoder

import torch.nn as nn
import torch


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.attn1 = MultiHeadAttention(d_model, heads)
        self.attn2 = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):

        attn_output1 = self.attn1(x, x, x, trg_mask)
        attn_output1 = self.dropout1(attn_output1)
        x = x + attn_output1
        x = self.norm1(x)

        attn_output2 = self.attn2(x, e_outputs, e_outputs, src_mask)
        attn_output2 = self.dropout2(attn_output2)
        x = x + attn_output2
        x = self.norm2(x)
        ff_output = self.ff(x)
        ff_output = self.dropout3(ff_output)
        x = ff_output + x
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionEncoder(d_model,100)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,trg,e_outputs,src_mask,trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x,e_outputs,src_mask,trg_mask)
        return self.norm(x)