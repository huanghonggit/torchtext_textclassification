import pandas as pd
import numpy as np
import torch
import time
import random
import os
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from argument import *


tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=64) #
LABEL = data.Field(sequential=False, use_vocab=False)
train_path = 'data/train.csv'
valid_path = "data/valid.csv"
test_path = "data/test.csv"



# 定义Dataset
class MyDataset(data.Dataset):

    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("texts", text_field), ("labels", label_field)]

        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['texts']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['texts'], csv_data['labels'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.2:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([text, label], fields))

        # 之前是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        # super(MyDataset, self).__init__(examples, fields, **kwargs)
        super(MyDataset, self).__init__(examples, fields)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.2): #
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


class LSTM(nn.Module):

    def __init__(self, weight_matrix):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2
        return y


def data_iter(train_path, valid_path, test_path, TEXT, LABEL):
    train = MyDataset(train_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False, aug=1)
    # 因为test没有label,需要指定label_field为None
    test = MyDataset(test_path, text_field=TEXT, label_field=None, test=True, aug=1)

    TEXT.build_vocab(train)
    weight_matrix = TEXT.vocab.vectors
    # 若只针对训练集构造迭代器
    # train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),  # 构建数据集所需的数据集
        batch_sizes=(16, 16),
        # 如果使用gpu，此处将-1更换为GPU的编号
        device=-1,
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.comment_text),
        sort_within_batch=False,
        repeat=False
    )
    test_iter = Iterator(test, batch_size=16, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, val_iter, test_iter, weight_matrix


def main():
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)

    model = LSTM(weight_matrix)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_funtion = F.cross_entropy

    # train
    for epoch in range(1, args.epochs + 1):
        for batch in (train_iter):
            steps = 0
            best_acc = 0
            last_step = 0
            feature, target = batch.texts, batch.labels
            # feature.data.t_(), target.data.sub_(1)
            optimizer.zero_grad()
            predicted = model(feature)# comment_text

            loss = loss_funtion(predicted, target)
            loss.backward()
            optimizer.step()
            print(loss)
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(predicted, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                print(train_acc)
                sys.stdout.write(
                    '\rBatch[{} / {}] - steps:{} - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch, args.epochs,
                                                                                             steps, loss.item(),
                                                                                             train_acc, corrects,
                                                                                             batch.batch_size))
            # 每100个batch用dev验证模型和参数
            if steps % args.test_interval == 0:
                dev_acc = eval(val_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        args.save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt

        # test
        model.eval()
        corrects, avg_loss = 0, 0
        for batch in (test_iter):
            feature, target = batch.texts, batch.labels
            feature.data.t_(), target.data.sub_(1)  #
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()
            corrects += (torch.max(logits, 1)
                         [1].view(target.size()).data == target.data).sum()
        size = len(test_iter.dataset)
        avg_loss /= size
        accuracy = 100.0 * corrects / size
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                           accuracy,
                                                                           corrects,
                                                                           size))

    torch.save(model.state_dict(), 'model/torchtext_LSTM_model.pth')
    # save model
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # save_prefix = os.path.join(save_dir, save_prefix)
    # save_path = '{}_steps_{}.pt'.format(save_prefix, steps)



if __name__ == '__main__':
    main()




#     model.eval()
#     eval_loss = 0
#     eval_acc = 0
#     for i, data in enumerate(testDataLoader):
#         x, y = data
#         if use_cuda:
#             x = Variable(x, volatile=True).cuda()
#             y = Variable(y, volatile=True).cuda()
#         else:
#             x = Variable(x, volatile=True)
#             y = Variable(y, volatile=True)
#         out = model(x)
#         loss = criterion(out, y)
#         eval_loss += loss.data * len(y)
#         _, pre = torch.max(out, 1)
#         num_acc = (pre == y).sum().item()
#         eval_acc += num_acc
#     print('test loss is:{:.6f},test acc is:{:.6f}'.format(
#         eval_loss / (len(testDataLoader) * batch_size),
#         eval_acc / (len(testDataLoader) * batch_size)))
#     if best_acc<(eval_acc / (len(testDataLoader) * batch_size)):
#         best_acc=eval_acc / (len(testDataLoader) * batch_size)
#         best_model=model.state_dict()
#         # print(best_model)
#         print('best acc is {:.6f},best model is changed'.format(best_acc))
#
# torch.save(model.state_dict(),'model/LSTM_model.pth')


















