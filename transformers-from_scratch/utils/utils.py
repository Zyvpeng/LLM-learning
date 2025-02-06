import numpy as np
from nltk.translate.bleu_score import  sentence_bleu,corpus_bleu,SmoothingFunction


def seq_padding(batch_data,padding=1):

    lens = [len(_) for _ in batch_data]

    max_len = max(lens)

    return np.array([
        np.concatenate([x,[padding]*(max_len-len(x))])if len(x)<max_len else x for x in batch_data
    ])


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