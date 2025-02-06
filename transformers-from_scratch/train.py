import torch

from utils.pre_data import PrepareData
from torch.nn import CrossEntropyLoss
from transformer import Transformer
from torch.optim import Adam
from tqdm import tqdm
def train(data, model, criterion, optimizer):
    for epoch in range(100):
        model.train()
        progress_bar = tqdm(data.train_data)
        for i,batch in enumerate(progress_bar):
            out = model(batch.src,batch.trg,batch.src_mask,batch.trg_mask)
            out = out.reshape(-1,data.cn_total_words)
            trg = batch.trg_y.reshape(-1)
            loss = criterion(out,trg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%50 ==0:
                total_dev_loss = 0
                n_tokens = 0
                with torch.no_grad():
                    for dev_batch in data.dev_data:
                        dev_out = model(dev_batch.src,dev_batch.trg,dev_batch.src_mask,dev_batch.trg_mask)
                        dev_l = criterion(dev_out.reshape(-1,data.cn_total_words),dev_batch.trg_y.reshape(-1))
                        total_dev_loss+=dev_l
                        n_tokens+=dev_batch.ntokens
                print(total_dev_loss/n_tokens)




if __name__ =='__main__':
    print('处理数据中')
    data = PrepareData("./data/train.txt","./data/dev.txt")
    model = Transformer(data.en_total_words,data.cn_total_words,512,32,8,0.1)
    total_para = 0
    for p in model.parameters():
        total_para+=p.numel()
    print(f'模型参数量为:{total_para/100000000}B')
    criterion = CrossEntropyLoss(ignore_index=1,reduction='sum')
    optimizer = Adam(lr=1e-4,params=model.parameters())
    train(data,model,criterion,optimizer)
