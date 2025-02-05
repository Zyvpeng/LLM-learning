import numpy as np


def seq_padding(batch_data,padding=0):

    lens = [len(_) for _ in batch_data]
    max_len = max(lens)

    return np.array([
        np.concatenate([x,[padding]*(max_len-len(x))])if len(x)<max_len else x for x in batch_data
    ])
