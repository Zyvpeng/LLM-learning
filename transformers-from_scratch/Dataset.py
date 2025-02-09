import torch
from torch.utils.data import Dataset

class En2CnDataset(Dataset):
    def __init__(self,train_file,dev_file,test_file):
        super().__init__()
    