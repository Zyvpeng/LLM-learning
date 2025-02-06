from Encoder import Encoder
from Decoder import Decoder

import torch
import torch.nn as nn

# 参考 https://github.com/taoztw/Transformer/blob/master/train.py
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

#
# src = torch.randint(0,1000,(8,30))
# trg = torch.randint(0,1000,(8,30))
# src_mask = torch.ones(8,30,30)
# trg_mask = torch.ones(8,30,30)
# trg_mask = torch.tril(trg_mask,0)
#
# model = Transformer(src_vocab=1000,trg_vocab=1000,d_model=512,N=10,heads=8,dropout=0.1)
# model(src,trg,src_mask,trg_mask)
