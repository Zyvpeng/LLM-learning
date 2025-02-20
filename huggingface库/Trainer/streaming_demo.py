import random
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import Trainer, TrainingArguments

class MyIterableDataset(IterableDataset):
    def __init__(self, data_num):
        self.data_num = data_num

    def __iter__(self):
        for _ in range(self.data_num):
            length = random.randint(1, 10)
            inputs = torch.tensor([random.randint(1, 100) for _ in range(length)])
            labels = torch.ones(length)
            print(inputs)
            yield {"inputs": inputs, "labels": labels}

class MyCollateFn:
    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, batch):
        max_length = max(len(x['inputs']) for x in batch)
        inputs = [self.pad_tensor(x['inputs'], max_length) for x in batch]
        labels = [self.pad_tensor(x['labels'], max_length) for x in batch]
        return {"inputs": torch.stack(inputs), "labels": torch.stack(labels)}

    def pad_tensor(self, tensor, length):
        return torch.cat([tensor, torch.full((length - tensor.size(0),), self.padding)])

class MyModel(nn.Module):
    def __init__(self, vocab_size, d_model, padding):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding)

    def forward(self, inputs, labels):
        embeddings = self.embedding(inputs)
        logits = self.linear(embeddings)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).long()
        loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# 实例化数据集和模型
dataset = MyIterableDataset(data_num=200)
collate_fn = MyCollateFn(padding=0)
model = MyModel(vocab_size=1000, d_model=512, padding=0)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./",
    num_train_epochs=10,
    per_device_train_batch_size=10,
    learning_rate=1e-3,
    logging_steps=10,
    max_steps = 10000,
    dataloader_num_workers=8
)

# 实例化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=dataset
)

# 开始训练
if __name__=='__main__':
    trainer.train()
