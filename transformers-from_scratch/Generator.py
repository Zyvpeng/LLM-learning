from transformer import Transformer
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self,model:Transformer,
                 input_vocab,
                 output_vocab,
                 input_index_dict,
                 output_index_dict,
                 decode_strategy="greedy",):
        super().__init__()
        self.model = model
        self.decode_strategy = decode_strategy
        self.input_vocab = input_vocab
        self.input_index_dict = input_index_dict
        self.output_vocab = output_vocab
        self.output_index_dict = output_index_dict

    # input: (seq_length)
    def forward(self,inputs):

        inputs = [self.input_vocab.get(word,0) for word in inputs]
        encoder_output = self.model.encoder(inputs)
        self.decode(encoder_output)

    def decode(self,encoder_output):
        # transformer decoder 逻辑如下
        # def forward(self, trg, e_outputs, src_mask, trg_mask):
        if self.decode_strategy=="greedy":
            decoder_input = torch.tensor([self.output_vocab["BOS"]]).unsqueeze(0)
            decoder_output = None
            for i in range(100):
                #生成一个下三角（包括对角线）为1的tensor
                output_len = decoder_input.size(-1)
                trg_mask = torch.ones((output_len,output_len))
                trg_mask = torch.triu(trg_mask,diagonal=1)
                trg_mask = (trg_mask==0)
                #(1,seq_len,output_vocab_size)
                decoder_output = self.model.decoder(decoder_input,encoder_output,None,trg_mask)
                ids = torch.argmax(decoder_output,dim=-1)
                id = ids[0,-1]
                if id.item()==self.output_vocab["EOS"]:
                    break
                decoder_input = torch.concat((decoder_output,id.unsqueeze(0).unsqueeze(0)),dim=-1)
            return decoder_output

x = torch.tensor([1]).unsqueeze(0)
print(x[0,-1])

