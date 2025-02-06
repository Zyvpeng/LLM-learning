import torch
from nltk import word_tokenize
from collections import Counter
import torch.nn.functional as F
import numpy as np
from .utils import seq_padding
class PrepareData:
    def __init__(self,train_file,dev_file):
        self.train_en ,self.train_cn = self.load_data(train_file)
        self.dev_en,self.dev_cn = self.load_data(dev_file)

        self.en_word_dict, self.en_total_words,self.en_index_dict = self.build_vocab(self.train_en)

        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_vocab(self.train_cn)

        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn,self.en_word_dict,self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        # print(self.train_en)
        #处理为batch数据
        self.train_data = self.splitToBatch(self.train_en,self.train_cn,16,True)
        self.dev_data = self.splitToBatch(self.dev_en,self.dev_cn,16,True)
    def load_data(self,path):
        en = []
        cn = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"]+word_tokenize(line[0].lower())+["EOS"])
                cn.append(["BOS"]+word_tokenize(" ".join([w for w in line[1]]))+["EOS"])
        return en,cn

    def build_vocab(self,sentences,max_words=50000):
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s]+=1


        #只用最常见的max_words构建字典
        ls = word_count.most_common(max_words)

        #加入unknown和pad两个token
        total_words = len(ls)+2
        word_dict = {w[0]:index+2 for index,w in enumerate(ls)}
        word_dict['UNK'] = 0
        word_dict['PAD'] = 1
        index_dict = {v:k for k,v in word_dict.items()}
        return word_dict,total_words,index_dict

    def wordToID(self,en,cn,en_dict,cn_dict):
        #不在vocab中的token的id设置为UNK
        out_en_ids = [[en_dict.get(w,0) for w in sent]for sent in en]
        out_cn_ids = [[cn_dict.get(w,0) for w in sent]for sent in cn]

        return out_en_ids,out_cn_ids

    def splitToBatch(self,en,cn,batch_size,shuffle=True):
        idx_list = np.arange(0,len(en),batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_idxs = []
        for idx in idx_list:
            batch_idxs.append(np.arange(idx,min(idx+batch_size,len(en))))

        batches = []

        for batch_idx in batch_idxs:
            batch_en = [en[index] for index in batch_idx]
            batch_cn = [cn[index] for index in batch_idx]

            batch_en = seq_padding(batch_en)
            batch_cn = seq_padding(batch_cn)
            batches.append(Batch(batch_en,batch_cn))
        return batches
class Batch:
    def __init__(self,src,trg,pad=1):
        src = torch.from_numpy(src).long()
        trg = torch.from_numpy(trg).long()
        self.src = src
        # src: batch_size,1,seq_len          再masked fill时，会广播为batch_size,seq_len,seq_len
        self.src_mask = (src!=pad).unsqueeze(-2)
        self.trg = trg[:,:-1]
        self.trg_y = trg[:,1:]
        # (batch_size,1,seq_length)
        self.trg_mask = (self.trg!=pad).unsqueeze(-2)
        trg_casual_mask = np.ones((1,self.trg_mask.size(-1),self.trg_mask.size(-1)))
        #左下角，包括对角线，为0
        trg_casual_mask = np.triu(trg_casual_mask,k=1)
        # (batch_size,1,seq_len) & (1,seq_len,seq_len) -> (batch_size,seq_len,seq_len)
        self.trg_mask = self.trg_mask&torch.from_numpy(trg_casual_mask==0)
        self.ntokens = (self.trg_y!=pad).sum()

# torch.cuda.is_available()
#
# src_mask = torch.tensor([[True,True,False],[True,False,False]])
# src_mask = src_mask.unsqueeze(-2)
# print(src_mask.shape)
# src = np.array([[2,2,3,4,5],[2,2,3,4,5],[1,2,1,1,1]])
# # src = torch.tensor([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])
# trg = src
#
# b = Batch(src,trg)
# print(b.trg)
# preds = torch.randn(3,4,10).view(-1,10)
# loss = torch.nn.CrossEntropyLoss(ignore_index=1,reduction='none')
# print(b.trg.contiguous().view(-1).shape)
# print(preds.shape)
# print(loss(preds,b.trg.contiguous().view(-1)))
# print(src.masked_fill(src_mask,-10))
# lines = "你好, 我是 郑屿蓬"
# lines = lines.strip()
# print(lines)
# print(word_tokenize(lines))
# x = {'a':-1}
# print(x)
# y = x.get("a",10)
# print(y)