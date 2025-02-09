from transformer import Transformer
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self,model:Transformer, decode_strategy="greedy"):
        super().__init__()
        self.model = model
        self.decode_strategy = decode_strategy

    def decode(self,inputs):
        self.model.decoder
