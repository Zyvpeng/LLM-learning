import random

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import Trainer,TrainingArguments

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

class my_collate_cls:
    def __init__(self,padding=0):
        self.padding = padding

    def __call__(self, batch_data):
        l = [x['inputs'].size(0) for x in batch_data]
        ML = max(l)
        inputs = []
        labels = []
        for data in (batch_data):
            input = data["inputs"]
            label = data["labels"]
            inputs.append(self.seq_padding(input, ML))
            labels.append(self.seq_padding(label, ML))
        # torch.stack():在新维度拼接list的tensor，前提是它们shape一样
        return {"inputs": torch.stack(inputs), "labels": torch.stack(labels)}

    def seq_padding(self,data, ml):
        if data.size(0) == ml:
            return data
        data = torch.concat((data, torch.tensor([self.padding] * (ml - data.size(0)))), dim=-1)
        return data

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

class myModel(nn.Module):
    def __init__(self,vocab_size,d_model,padding):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size,d_model)
        self.out = nn.Linear(d_model,vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding)
        print("init")

    def forward(self,inputs,labels):
        #batchsize, seq_length, d_model
        x = self.embed(inputs)
        out = self.out(x)
        out = out.reshape(-1,self.vocab_size)
        labels =labels.reshape(-1).long()
        loss = self.loss_fn(out,labels)

        #loss是为了满足trainer的compute_loss函数的传参要求
        #compute_loss 用 data["loss"]的方法获取loss，进行反向传播的参数更新
        #logits这个key是为了将来进行推力时用
        return {"loss":loss,"logits":out}


dataset = Mydataset(200)
#torch的dataloader可以使用class或func作为collate_fn
# dataloader = DataLoader(dataset,collate_fn=my_collate_cls(0),shuffle=True,batch_size=4)
# for i in dataloader:
#     print(i)
#     print(123)
# print(data[1])
collate_fn=my_collate_cls(0)
model = myModel(1000,512,0)
training_args = TrainingArguments(
    num_train_epochs=10,
    output_dir="./",
    per_device_train_batch_size=10,
    learning_rate=1e-3,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset
)
trainer.train()

