import numpy as np
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

#每句话最大长度
MAX_LEN=64
#数据清理
def clean(sent):
    punctuation_remove = u'[、：，？！。；……（）『』《》【】～!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    sent = re.sub(r'ldquo', "", sent)
    sent = re.sub(r'hellip', "", sent)
    sent = re.sub(r'rdquo', "", sent)
    sent = re.sub(r'yen', "", sent)
    sent = re.sub(r'⑦', "7", sent)
    sent = re.sub(r'(， ){2,}', "", sent)
    sent = re.sub(r'(！ ){2,}', "", sent)  # delete too many！，？，。等
    sent = re.sub(r'(？ ){2,}', "", sent)
    sent = re.sub(r'(。 ){2,}', "", sent)
    # sent=sent.split()
    # sent_no_pun=[]
    # for word in sent:
    #     if(word!=('，'or'。'or'？'or'！'or'：'or'；'or'（'or'）'or'『'or'』'or'《'or'》'or'【'or'】'or'～'or'!'or'\"'or'\''or'?'or','or'.')):
    #         sent_no_pun.append(word)
    # s=' '.join(sent_no_pun)
    sent = re.sub(punctuation_remove, "", sent)  # delete punctuations
    #若长度大于64，则截取前64个长度
    if(len(sent.split())>MAX_LEN):
        s=' '.join(sent.split()[:MAX_LEN])
    else:
        s = ' '.join(sent.split())  # delete additional space

    return s
#创建词典
word_to_inx={'pad':0}
inx_to_word={0:'pad'}
#获取数据，并转换为id
def get_data():
    test_data = open('.data/phone_test.txt', 'r', encoding='utf-8').readlines()
    test_data = [clean(line).replace('\n', '').replace('\r', '') for line in test_data]
    print(test_data)
    test_data_label=[0 for i in range(len(test_data))]
    print(test_data)

    train_data = open('.data/phone_train.txt.txt', 'r', encoding='utf-8').readlines()
    train_data = [clean(line).replace('\n', '').replace('\r', '') for line in train_data]
    train_data_label = [1 for i in range(len(train_data))]

    dev_data = open('.data/phone_dev.txt.txt', 'r', encoding='utf-8').readlines()
    dev_data = [clean(line).replace('\n', '').replace('\r', '') for line in dev_data]
    dev_data_label = [2 for i in range(len(dev_data))]

    # 合并所有文本和标签
    data = test_data + train_data + dev_data
    print(len(data))
    data_label = test_data_label+dev_data_label+train_data_label
    # print(data[0:5])
    # print(data_label[0:5])

    # 对所有的不重复的词汇进行字典标号
    vocab = [word for s in data for word in s.split()]
    vocab = set(vocab)
    #print(vocab)
    for word in vocab:
        inx_to_word[len(word_to_inx)]=word
        word_to_inx[word]=len(word_to_inx)

    # 对数据集进行字典编号
    data_id=[]
    for s in data:
        s_id=[]
        for word in s.split():
            s_id.append(word_to_inx[word])
        s_id=s_id+[0]*(MAX_LEN-len(s_id))
        data_id.append(s_id)
    # 求出句子的最大长度
    # max_len=0
    # for s in data_id:
    #     if len(s)>max_len:
    #         max_len=len(s)
    # print(max_len)
    # 318
    # print(data_id[5:10])
    # print(len(data_id),' ',len(data_label))
    return data_id, data_label, word_to_inx, inx_to_word
# 获取两个字典

def get_dic():
    _,_,word_to_inx,inx_to_word = get_data()
    return word_to_inx, inx_to_word

# 将数据转化为tensor
def tensorFromData():
    data_id, data_lable, _, _ = get_data()
    data_id_train, data_id_test, data_label_train, data_label_test = train_test_split(data_id, data_lable, test_size=0.2, random_state=20190409)

    data_id_train = torch.LongTensor(data_id_train)  # 为什么 用longtensor
    data_id_test = torch.LongTensor(data_id_test)
    data_label_train = torch.LongTensor(data_label_train)
    data_label_test = torch.LongTensor(data_label_test)

    return data_id_train, data_id_test, data_label_train, data_label_test
    # return data_id,data_lable

class TextDataSet(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]