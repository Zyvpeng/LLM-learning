import torch.nn as nn
import  torch
import math
class PositionEncoder(nn.Module):
    def __init__(self,d_model,max_seq_length):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_length,d_model)
        for pos in range(max_seq_length):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))

        #(1,max_seq_length,d_model)
        #记录到模型参数中,存入state_dict() 可以随着to(device)移动设备，唯一区别是不计算梯度，不反向传播
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)



    #(batchsize,seq_length,d_model)
    def forward(self,x):
        x = x*math.sqrt(self.d_model)
        seq_length = x.size(1)
        # print(self.pe.requires_grad)
        x = x + self.pe[:,:seq_length]
        return x
# model = PositionEncoder(512,100)
# print(model.state_dict())
# x = torch.randn(8,30,512)
# model(x)
