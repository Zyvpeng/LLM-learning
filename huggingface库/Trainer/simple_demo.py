import random

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

class Mydataset(Dataset):
    def __init__(self,data_num):
        self.inputs = []
        self.labels = []
        for _ in range(data_num):
            d = [ random.randint(1,100) for i in range(random.randint(1,10))]
            d = torch.tensor(d)
            self.inputs.append(d)
            label = torch.ones(d.size(0))
            self.labels.append(label)
            print(label.shape)
            print(d.shape)
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {"inputs":self.inputs[i],"labels":self.labels[i]}

def my_collate_fn(batch_data,padding=0):
    def seq_padding(data,ml,padding=0):
        if data.size(0) ==ml:
            return data
        data = torch.concat((data,torch.tensor([padding]*(ml-data.size(0)))),dim=-1)
        return data
    l  = [x['inputs'].size(0) for x in batch_data]
    ML = max(l)
    inputs = []
    labels = []
    for data in (batch_data):
        input =  data["inputs"]
        label = data["labels"]
        inputs.append(seq_padding(input,ML))
        labels.append(seq_padding(label,ML))
    #torch.stack():在新维度拼接list的tensor，前提是它们shape一样
    return {"inputs":torch.stack(inputs),"labels":torch.stack(labels)}



dataset = Mydataset(200)
dataloader = DataLoader(dataset,collate_fn=my_collate_fn,shuffle=True,batch_size=4)
for i in dataloader:
    print(i)
    print(123)
# print(data[1])
