# Stream Dataset下训练方法

最近在进行VLM压缩模型训练时，遇到了一个问题，就是用到的500k数据集规模太大了，没办法直接加载到内存中处理

还有一个更严重的问题，就是数据进行map预处理后，甚至没办法在硬盘中存放

---



## Stream Dataset

为了解决上述问题，我了解到了一种流式加载数据的方案，也就是每次只在需要时（训练要用到）取数据进行处理，然后释放

- 在原生pytorch中，可以进行如下定义

```python
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
```

这里的data_num就是数据的个数，\__iter__方法会在dataset被迭代是调用，

每次只返回yield {"inputs": inputs, "labels": labels}这一部分数据

- 在transformer的dataset库中，可以进行如下定义

如果要从huggingface格式的数据集加载，则可以用如下方法

```python
load_dataset
```

它的定义如下

```python
def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    trust_remote_code: bool = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
```

其中有几个超参数值得关注

1. keep_in_memory：是否加载入内存中，默认不加载。如果数据集很大，且无法完全加载到内存中，或者需要按需处理数据（如流式加载），`keep_in_memory=False` 是一个合适的选择
2. num_proc:处理数据的进程数，大一点会加速处理
3. ==streaming==：默认为False，当设置为True时，会流式加载数据，也就是用到数据时才加载进来，适用于超大数据集或者想要快速查看下数据时

如果从自己保存的arrow数据集（save_to_disk保存的）加载，则可以用

```python
load_from_disk
```

它的定义如下

```python
def load_from_disk(
    dataset_path: PathLike, keep_in_memory: Optional[bool] = None, storage_options: Optional[dict] = None
) -> Union[Dataset, DatasetDict]:
```

然后需要把Dataset 或 DatasetDict 转为IterableDataset或IterableDatasetDic

方法很简单

```python
train_dataset = train_dataset.to_iterable_dataset(num_shards=8)
```

**num_shards很关键**

- `num_shards` 控制将数据集分成多少个分片（shards）。这在分布式训练或多进程加载数据时非常有用。
- 每个分片（shard）包含数据集的一部分，通常是在分布式训练时，多个进程（或机器）将处理不同的数据分片，从而加速数据加载。

---



## map方法

往往需要对dataset进行map预处理

例如

```python
train_dataset = train_dataset.map(instruct_ft_tokenize_qwen_fuction, batched=True, batch_size=512, fn_kwargs={"model": model, "mem": MEM_TOKENS})
```

map定义如下

```python
  def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":
```

可以发现默认也是会在磁盘中保存的（==我遇到的问题是磁盘中存不下了==）

**num_proc可以加速处理**

## Trainer训练

要想进行流式训练，有两种方案

1. 流式数据集+正常map处理

   因为数据是一点一点取出来的，所以map处理自然也就是一点一点进行的，不会带来磁盘压力

   参考：https://huggingface.co/docs/datasets/v1.10.1/dataset_streaming.html

2. 正常数据集+with_transform处理

   ```python
       def with_transform(
           self,
           transform: Optional[Callable],
           columns: Optional[List] = None,
           output_all_columns: bool = False,
       ):
           """Set `__getitem__` return format using this transform. The transform is applied on-the-fly on batches when `__getitem__` is called.
   
           As [`~datasets.Dataset.set_format`], this can be reset using [`~datasets.Dataset.reset_format`].
   
           Contrary to [`~datasets.Dataset.set_transform`], `with_transform` returns a new [`Dataset`] object.
   
           Args:
               transform (`Callable`, `optional`):
                   User-defined formatting transform, replaces the format defined by [`~datasets.Dataset.set_format`].
                   A formatting function is a callable that takes a batch (as a `dict`) as input and returns a batch.
                   This function is applied right before returning the objects in `__getitem__`.
               columns (`List[str]`, `optional`):
                   Columns to format in the output.
                   If specified, then the input batch of the transform only contains those columns.
               output_all_columns (`bool`, defaults to `False`):
                   Keep un-formatted columns as well in the output (as python objects).
                   If set to `True`, then the other un-formatted columns are kept with the output of the transform.
   ```

   与map方法不同，with_transform只有在\__getitem__时会被调用，也可以起到流处理的作用



### 方法1

- max_steps设置

因为IterableDataset不具备len属性，所以trainer并不知道数据集有多大（很简单，数据集都没有load呢， 怎么可能知道大小呢）

所以要只能需要训练的次数

```python
training_args = TrainingArguments(
    output_dir="./",
    num_train_epochs=10,
    per_device_train_batch_size=10,
    learning_rate=1e-3,
    logging_steps=10,
    max_steps = 10000,
    dataloader_num_workers=8
)
```

max_steps的计算为:**max_steps = (num_samples // batch_size) // gradient_accumulation_steps * epochs**

这里参考了：[HuggingFace Trainer max_step to set for streaming dataset - Stack Overflow](https://stackoverflow.com/questions/76011298/huggingface-trainer-max-step-to-set-for-streaming-dataset)

如果不想支持多进程的data_loader，到这里就可以了

但是要想实现高效训练，避免gpu、cpu频繁切换，还是要实现多进程的

- 多进程支持

方法很简单

```python
training_args.dataloader_num_workers=8
```

这在非流式训练中可以直接适配，不过在流式中需要做一些调整

1. 用main检查包裹train

   ```python
   if __name__=='__main__':
       trainer.train()
   ```

2. 保证dataloader_num_workers<=dataset.num_shards

   1. 在load_dataset加载的数据集中，num_shards等于arrow文件个数

   2. train_dataset = train_dataset.to_iterable_dataset(num_shards=8),disk数据集转换情况下可以手动设置

      参考：https://discuss.huggingface.co/t/num-worker-with-iterabledataset/58914

可以看到，流式训练的效率不会比一次性加载差很多

![image-20250221003758696](https://s2.loli.net/2025/02/21/ral3MHZ6IGv7cqU.png)



- 每个epoch shuffle
  - 参考这里https://huggingface.co/docs/datasets/stream
  - trainer训练时会调用set_epoch方法改变种子
  - 只需要训练开始时调用**dataset.shuffle(seed=xx,buffer_size=xx)**就好了，这会使得dataset每次从stream中取出buffer_size条数据，然后打乱

### 方法2

还没有试过





