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

def get_label(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            result_labels = list()
            for line in f.readlines():
                result_labels.append(int(line.split('\t')[0]))
            return result_labels

#获取数据，并转换为id
def get_data():
    train_data = open('./data/train.txt', 'r', encoding='GBK').readlines()
    train_data = [clean(line).replace('\n', '') for line in train_data]
    train_labels = get_label('./data/phone_train.txt')

    test_data = open('data/test.txt', 'r', encoding='GBK').readlines()
    test_data = [clean(line).replace('\n', '') for line in test_data]
    test_labels = get_label('./data/phone_test.txt')

    dev_data = open('data/valid.txt', 'r', encoding='GBK').readlines()
    dev_data = [clean(line).replace('\n', '') for line in dev_data]
    dev_labels = get_label('./data/phone_dev.txt')

    data = train_data + test_data + dev_data
    data_label = train_labels + test_labels + dev_labels
    # print(data[0:5])
    # print(data_label[0:5])
    vocab = [word for s in data for word in s.split()]
    vocab = set(vocab)   # 收集所有不重复的词
    #print(vocab)

    word_to_inx = {word: i for i, word in enumerate(vocab)}   # 对每个word编号
    inx_to_word = {word_to_inx[word]: word for word in word_to_inx}  # 对每个编号赋给word

    data_id = []
    for s in data:  #把每句话中的词编号
        s_id = []
        for word in s.split():
            s_id.append(word_to_inx[word])
        s_id = s_id+[0]*(MAX_LEN-len(s_id))    # 句子长度不够64维则补0
        data_id.append(s_id)

    return data_id, data_label, word_to_inx, inx_to_word

# 获取两个字典     这里get_data()函数调用了两次
# def get_dic():
#     _,_,word_to_inx,inx_to_word=get_data()
#     return word_to_inx,inx_to_word

# 将数据转化为tensor
def tensorFromData():
    data_id, data_lable, word_to_inx, inx_to_word=get_data()
    data_id_train, data_id_test, data_label_train, data_label_test = train_test_split(data_id, data_lable, test_size=0.2, random_state=20190410)

    data_id_train = torch.LongTensor(data_id_train) # 将数据转成张量格式
    data_id_test = torch.LongTensor(data_id_test)
    data_label_train = torch.LongTensor(data_label_train)
    data_label_test = torch.LongTensor(data_label_test)
    return data_id_train, data_id_test, data_label_train, data_label_test, word_to_inx, inx_to_word

class TextDataSet(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]















    # def vocab(data):
    #     # 对不重复的词，做索引词典
    #     # vocab = [word for s in data for word in s.split()]
    #     #     # vocab = set(vocab)
    #     #     # print(len(vocab))
    #     corpus = data
    #     vector = TfidfVectorizer()
    #     tfidf = vector.fit_transform(corpus)
    #     print(tfidf)
    #     return tfidf
    #     # vectorizer = CountVectorizer()
    #     # print(vectorizer.fit_transform(data).toarray())
    #     # print(vectorizer.get_feature_names())
    #
    #
    # def data_to_vectors(filename, save_filename):
    #     with open(filename, 'r', encoding='utf-8') as f:
    #         result_data = list()
    #         result_labels = list()
    #         for line in f.readlines():
    #             result_data.append(line.replace('\n', '').split('\t')[1])
    #             result_labels.append(int(line.split('\t')[0]))
    #
    #         open(save_filename, 'w').write('%s' % '\n'.join(result_data))
    #
    #         return result_data, result_labels
    #
    #
    # # 获取数据，并转换为id
    # def get_data():
    #     data_train, label_train = data_to_vectors('data/phone_train.txt', 'data/train.txt')
    #     data_test, label_test = data_to_vectors('data/phone_test.txt', 'data/test.txt')
    #     data_valid, label_valid = data_to_vectors('data/phone_dev.txt', 'data/valid.txt')
    #
    #
    #     train_tfidf = vocab(data_train)
    #     test_tfidf = vocab(data_test)
    #
    #     return train_tfidf, label_train, test_tfidf, label_test     #, word_to_inx, inx_to_word

    # # 对数据集进行字典编号
    # def data_to_ID(data):
    #     data_id = []
    #     for s in data:
    #         s_id = []
    #         for word in s.split():
    #             s_id.append(word_to_inx[word])
    #         s_id = s_id + [0] * (MAX_LEN - len(s_id))
    #         data_id.append(s_id)
    #     return data_id


    # def get_data():
    #     data_train, label_train = data_to_vectors('data/phone_train.txt', 'data/train.txt')
    #     data_test, label_test = data_to_vectors('data/phone_test.txt', 'data/test.txt')
    #
    #     train_tfidf = vocab(data_train)
    #     test_tfidf = vocab(data_test)
    #
    #     return train_tfidf, label_train, test_tfidf, label_test     #, word_to_inx, inx_to_word








    # data_id, data_lable, _, _ = get_data()
    # data_id_train, data_id_test, data_label_train, data_label_test = train_test_split(data_id, data_lable,
    #                                                                                   test_size = 0.2,
    #                                                                                   random_state = 20190409)

    # data_label_train_new = []
    # for each in data_label_train:
    #     data_label_train_new.append(int(each))




    #test_data = open('data/phone_test.txt', 'r', encoding='utf-8').readlines()
    # test_data = [clean(line).replace('\n', '').split(' ')[1] for line in test_data]
    # test_data_label = [clean(line).replace('\n', '').split('\t')[0] for line in test_data]
    #
    # # 把str标签转成int
    # test_data_label_new = []
    # for each in test_data_label:
    #     test_data_label_new.append(int(each))
    #
    # # 汇总训练集和测试集数据
    # data = train_data + test_data
    # data_label = train_data_label_new + test_data_label_new
    #
    # print(len(data))
    # print(len(data_label))
    #
    # # 对汇总的数据组建词典
    # word_to_inx, inx_to_word = vocab(data)
    #
    # # 把训练集、测试集文字转序列
    # train_data_id = data_to_ID(train_data)
    # test_data_id = data_to_ID(test_data)


    # train_data = open('data/phone_train.txt', 'r', encoding='utf-8').readlines()
    # train_data = [clean(line).replace('\n', '').split(' ') for line in train_data]
    # train_data_label = [clean(line).replace('\n', '').split('\t')[0] for line in train_data]

    # 把str标签转成int
    # train_data_label_new = []
    # for each in train_data_label:
    #     train_data_label_new.append(int(each))

    #
    # # 数据清理
    # def clean(sent):
    #     punctuation_remove = u'[、：，？！。；……（）『』《》【】～!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    #     sent = re.sub(r'ldquo', "", sent)
    #     sent = re.sub(r'hellip', "", sent)
    #     sent = re.sub(r'rdquo', "", sent)
    #     sent = re.sub(r'yen', "", sent)
    #     sent = re.sub(r'⑦', "7", sent)
    #     sent = re.sub(r'(， ){2,}', "", sent)
    #     sent = re.sub(r'(！ ){2,}', "", sent)  # delete too many！，？，。等
    #     sent = re.sub(r'(？ ){2,}', "", sent)
    #     sent = re.sub(r'(。 ){2,}', "", sent)
    #     sent = re.sub(punctuation_remove, "", sent)  # delete punctuations
    #     # 若长度大于64，则截取前64个长度
    #     if (len(sent.split()) > MAX_LEN):
    #         s = ' '.join(sent.split()[:MAX_LEN])
    #     else:
    #         s = ' '.join(sent.split())  # delete additional space
    #
    #     return s
    #
    #
    # # 创建词典
    # word_to_inx = {'pad': 0}
    # inx_to_word = {0: 'pad'}
    #
    #
    # # 获取文件内容
    # def getData(file):
    #     f = open(file, 'r', encoding='utf-8')
    #     raw_data = f.readlines()
    #     return raw_data
    #
    # # 转换文件格式
    # def d2csv(raw_data, name):
    #     texts = []
    #     labels = []
    #     i = 0
    #     for line in raw_data:
    #         label = line.split('\t')[0]
    #         text = line.split('\t')[1]
    #         texts.append(text)
    #
    #         labels.append(label)
    #         i += 1
    #         # if i % 1000 == 0:
    #         #     print(i)
    #     df = pd.DataFrame({'texts': texts, 'labels': labels})
    #     print(df.shape)
    #     df.to_csv('data/' + name + '.csv', index=False)  # 保存文件
    #     # print(labels)
    #
    # #
    # # def vocab(data):
    # #     # 对不重复的词，做索引词典
    # #     vocab = [word for s in data for word in s.split()]
    # #     vocab = set(vocab)
    # #     print(len(vocab))
    # #     # print(len(vocab))
    # #     # print(vocab)
    # #     for word in vocab:
    # #         inx_to_word[len(word_to_inx)] = word
    # #         word_to_inx[word] = len(word_to_inx)
    # #
    # #     return  word_to_inx, inx_to_word
    #

    #
    #
    #     # print(len(vocab))
    #     # print(vocab)
    #     for word in vocab:
    #         inx_to_word[len(word_to_inx)] = word
    #         word_to_inx[word] = len(word_to_inx)
    #
    #     return  word_to_inx, inx_to_word









    # test_data = open('data/phone_test.txt', 'r', encoding='utf-8').readlines()
    # test_data = [clean(line).replace('\n', '').replace('\r', '').split('\t')[0] for line in test_data]
    # test_data_label = [0 for i in range(len(test_data))]
    #
    # train_data = open('data/phone_train.txt', 'r', encoding='utf-8').readlines()
    # train_data = [clean(line).replace('\n', '').replace('\r', '') for line in train_data]
    # train_data_label = [1 for i in range(len(train_data))]
    #
    # dev_data = open('data/phone_dev.txt', 'r', encoding='utf-8').readlines()
    # dev_data = [clean(line).replace('\n', '').replace('\r', '') for line in dev_data]
    # dev_data_label = [2 for i in range(len(dev_data))]

    # data = []
    # data_label = []

    # test_data = getData('data/phone_test.txt')
    # d2csv(test_data, 'test')





    # train_data = getData('data/phone_train.txt')
    # d2csv(train_data, 'train')

    # dev_data = getData('data/phone_dev.txt')
    # d2csv(dev_data, 'dev')

    # train_data = pd.read_csv('data/train.csv')
    # data.append(train_data.ix[:, 0])
    # data_label.append(train_data.ix[:, 1])

    # test_data = pd.read_csv('data/test.csv')
    # data.append(test_data.ix[:, 0])
    # data_label.append(test_data.ix[:, 1])
    #
    # dev_data = pd.read_csv('data/dev.csv')
    # data.append(dev_data.ix[:, 0])
    # data_label.append(dev_data.ix[:, 1])

    # print(data)
    # print(data_label_new)
    # data.append(train_data.ix[:, 0]) + test_data.ix[:, 0] + dev_data.ix[:, 0]
    # data_label = train_data.ix[:, 1] + test_data.ix[:, 1] + dev_data.ix[:, 1]

    # # 合并所有文本和标签
    # data = test_data + train_data + dev_data
    # print(len(data))
    # print(len(data_label))
    # data_label = test_data_label + dev_data_label + train_data_label
    # print(data[0:5])
    # print(data_label[0:5])

