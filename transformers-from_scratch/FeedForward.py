import torch
import torch.nn as nn
import torch.nn.functional as F
class FeedForward(nn.Module):
    def __init__(self,d_model=512,d_ff=2048,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
