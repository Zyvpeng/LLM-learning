import numpy as np
from nltk.translate.bleu_score import  sentence_bleu,corpus_bleu,SmoothingFunction
import torch

def seq_padding(batch_data,padding=1):

    lens = [len(_) for _ in batch_data]

    max_len = max(lens)

    return np.array([
        np.concatenate([x,[padding]*(max_len-len(x))])if len(x)<max_len else x for x in batch_data
    ])

def read_json(file_path):
    import json
    with open(file_path,'r') as f:
        data = [json.loads(line) for line in f]
        print(data)

def subsequent_mask(size):
    """
    deocer层self attention需要使用一个mask矩阵，
    :param size: 句子维度
    :return: 右上角(不含对角线)全为False，左下角全为True的mask矩阵
    """
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0
#计算单独句子的bleu
def compute_bleu(candidate,reference):
    # 平滑函数会对那些 0 计数的 n-gram 进行平滑处理，避免 BLEU 得分直接归零，从而更合理地反映翻译质量
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(reference,candidate,weights=(0.25,0.25,0,0),smoothing_function=smoothing)
    print(score)

#在一个语料库上计算整体的bleu
def compute_whole_bleu(candidate,reference):
    smoothing = SmoothingFunction().method1
    score = corpus_bleu(reference,candidate,weights=(0.25,0.25,0,0),smoothing_function=smoothing)
    print(score)

def test():
    reference = [['this','is','a','test'] ,['this','is','a','quiz']]
    candidate = ['this','is','test']
    compute_bleu(candidate,reference)

    reference = [[['this','is','a','test'] ,['this','is','a','quiz']],[['i','am','very','happy'],['i','am','very','glad']]]
    candidate = [['this','is','test'],['i','am','very','happy']]
    compute_whole_bleu(candidate,reference)