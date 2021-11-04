import torch
from torch import nn
from torch.utils import data as Data
from tokenizer import Tokenizer
from config import Config
import os
import numpy as np
from model import Transformer
from tqdm import tqdm


config = Config.from_json_file("config.json")

# 读取数据，创建token
def read_data():
    global token
    with open("./data/new_year.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    if os.path.exists("word_index.json"):
        token = Tokenizer.from_word_json("word_index.json")
    else:
        token = Tokenizer()
        token.fit_text(data, save_file=True)
    config.vocab_size = len(token.word_index) - 1
    return data

# 构建自己的数据集
class NewYearData(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sentence = self.data[index]
        up_sentence, down_sentence = sentence.split("|")
        up_sentence = token.encoder_sentence([up_sentence])
        up_sentence = token.padding(up_sentence)

        down_sentence = token.encoder_sentence([down_sentence])
        down_sentence = token.padding(down_sentence)

        return np.int32(up_sentence[0, :-1]), np.int32(down_sentence[0, :-1]), np.int64(down_sentence[0, 1:])

    def __len__(self):
        return len(self.data)

def train():
    data = read_data()
    train_data = NewYearData(data)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    # 创建网络
    model = Transformer(config.vocab_size, config.vocab_size, trg_pad_idx=0, src_pad_idx=0)
    model.train()

    # 初始化参数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fc = nn.CrossEntropyLoss(ignore_index=0)

    nb = len(train_data)

    old_loss_time = None
    old_loss = 10
    for epoch in range(1, config.epochs):
        pbar = tqdm(train_data, total=nb)
        optimizer.zero_grad()

        for step, (x, x_y, y) in enumerate(pbar):

            logits = model(x, x_y)

            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)

            loss = loss_fc(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if old_loss_time is None:
                loss_time = loss
                old_loss_time = loss
            else:
                old_loss_time += loss
                loss_time = old_loss_time / (step+1)

            s = ("train ===> epoch: {} ---- step: {} ---- loss: {:.4f} ---- loss_time: {:.4f}".format(epoch, step, loss, loss_time))
            pbar.set_description(s)

        if old_loss > loss_time and epoch > 2:
            torch.save(model.state_dict(), "transformer.pkl")


if __name__ == '__main__':
    train()