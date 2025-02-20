import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset,concatenate_datasets
from modeling_icae_multi_span_qwen2vl_2b_finetune import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import instruct_ft_tokenize_qwen_fuction, DataCollatorForVLMPadding, train_model
from datasets import Dataset

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
training_args.per_device_train_batch_size=8
training_args.per_device_eval_batch_size=8
training_args.gradient_accumulation_steps=1
training_args.dataloader_pin_memory=False,
training_args.logging_steps=100
training_args.save_steps = 5000
training_args.learning_rate=1e-4
training_args.num_train_epochs=2 
training_args.save_safetensors=False
training_args.eval_strategy = "steps"
training_args.eval_steps = 500
training_args.dataloader_num_workers=8
training_args.max_steps = ((500000-500) // 8) // training_args.gradient_accumulation_steps * training_args.num_train_epochs 
print(training_args)
print(model_args)
print(data_args)

# training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code
# d = load_dataset("merve/vqav2-small")
# print(d)
lora_config = LoraConfig(
    r=128,   #model_args.lora_r,
    lora_alpha=16,
    lora_dropout=0.02,
    inference_mode=False,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# check model_args.mem_size and min_tokens_for_lm
assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    

memory_size = training_args.fixed_mem_size



print("Loading dataset...")


model = ICAE(model_args, training_args, lora_config)
MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
data_list = []
for i in range(10):
    data_list.append(Dataset.load_from_disk(f'/root/autodl-tmp/vlm_compress/icae/code/icae_v2/utils/GQA500000/batch_{i}'))
d = concatenate_datasets(data_list)

train_dataset = d.select(range(500,500000))
eval_dataset = d.select(range(0,500))
# train_dataset = load_dataset("merve/vqav2-small")['validation'].select(range(1000,20000))
# eval_dataset = load_dataset("merve/vqav2-small")['validation'].select(range(0,100))
# train_dataset = Dataset.from_json("/root/autodl-tmp/vlm_compress/icae/code/icae_v2/data/finetune/data_vl.json")
# eval_dataset = Dataset.from_json("/root/autodl-tmp/vlm_compress/icae/code/icae_v2/data/finetune/data_vl.json")
train_dataset = train_dataset.to_iterable_dataset(num_shards=8)
eval_dataset = eval_dataset.to_iterable_dataset(num_shards=8)
train_dataset = train_dataset.map(instruct_ft_tokenize_qwen_fuction, batched=True, batch_size=512, fn_kwargs={"model": model, "mem": MEM_TOKENS})
eval_dataset = eval_dataset.map(instruct_ft_tokenize_qwen_fuction, batched=True, batch_size=64, fn_kwargs={"model": model, "mem": MEM_TOKENS})
data_collator = DataCollatorForVLMPadding(model.pad_token_id)

if __name__ =='__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    train_model(model, train_dataset, eval_dataset, training_args,data_collator)
