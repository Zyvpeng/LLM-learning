
import torch
# from torch.utils.data import IterableDataset
from datasets import IterableDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

data = [
    {
        "input_ids": torch.tensor([101, 2040, 2001, 1999, 14936, 102]),
        "token_type_ids": torch.tensor([0, 0, 0, 0, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
    },
    {
        "input_ids": torch.tensor([101, 2040, 102]),
        "token_type_ids": torch.tensor([0, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 1]),
    },
    {
        "input_ids": torch.tensor([101, 2040, 2001, 1999]),
        "token_type_ids": torch.tensor([0, 0, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 1]),
    },
    {
        "input_ids": torch.tensor([101, 2040, 2001, 1999, 14936, 102]),
        "token_type_ids": torch.tensor([0, 0, 0, 0, 0, 0]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
    },
    {
        "input_ids": torch.tensor([101]),
        "token_type_ids": torch.tensor([00]),
        "attention_mask": torch.tensor([1]),
    },
    {
        "input_ids": torch.tensor([101]),
        "token_type_ids": torch.tensor([00]),
        "attention_mask": torch.tensor([1]),
    },
]

def data_generator():
    for example in data :  # 数据扩展 20 倍
        yield example

# 使用 from_generator 方式创建 Hugging Face 的 IterableDataset
hf_dataset = IterableDataset.from_generator(data_generator)
#需要手动设置shuufle选项，并在每个epoch改变seed
hf_dataset =   hf_dataset.shuffle(seed=42,buffer_size=2)
#训练时要按照如下规范进行
for epoch in range(10):
    hf_dataset.set_epoch(epoch)
    for x in hf_dataset:
        print(x)

print(hf_dataset)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")
train_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    dataloader_num_workers=0,
    split_batches=True,
    max_steps=10000,
    # eval_strategy="epoch",
    # eval_on_start=True,
    # eval_steps = 1,
)
print(train_args)
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer)

trainer = Trainer(
    eval_dataset=hf_dataset,
    train_dataset=hf_dataset,
    model=model,
    args=train_args,
    data_collator=dc,
)
if __name__=='__main__':
    trainer.train()