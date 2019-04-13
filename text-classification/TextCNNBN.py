import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_preprocess
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda=torch.cuda.is_available()


# 将数据划分为训练集和测试集
X_train, X_test, Y_train, Y_test, word_to_inx, inx_to_word = data_preprocess.tensorFromData()
trainDataSet = data_preprocess.TextDataSet(X_train, Y_train)
testDataSet = data_preprocess.TextDataSet(X_test, Y_test)
trainDataLoader = DataLoader(trainDataSet, batch_size=16, shuffle=True)
testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False)

# 获取字典
# word_to_inx, inx_to_word = data_preprocess.get_dic()
word_to_inx, inx_to_word = word_to_inx, inx_to_word
len_dic = len(word_to_inx)

# 定义超参数
MAXLEN = 64
input_dim = MAXLEN
emb_dim = 128
num_epoches = 20
batch_size = 16
learning_rate = 0.001

#定义模型
class TextCNNBN_model(nn.Module):
    def __init__(self,len_dic,emb_dim,input_dim):
        super(TextCNNBN_model,self).__init__()
        self.embed=nn.Embedding(len_dic,emb_dim)#b,64,128
        self.conv1=nn.Sequential(
            nn.Conv1d(input_dim,256,1,1,padding=0),#b,256,128
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 1, 1, padding=0),  # b,128,128
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,128,64
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(input_dim,256,3,1,padding=1),#b,256,128
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 3, 1, padding=1),  # b,128,128
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,128,64
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(input_dim,256,5,1,padding=2),#b,256,128
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 5, 1, padding=2),  # b,128,128
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2,2)#b,128,64
        )
        #b,128,64+64+64
        #b,128*192
        self.drop=nn.Dropout(0.5)#b,128*192
        self.classify=nn.Linear(128*192,3)#b,3
    def forward(self, x):
        x=self.embed(x)
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=self.conv3(x)
        x=torch.cat((x1,x2,x3),2)
        b,c,d=x.size()
        x=x.view(-1,c*d)
        x=self.drop(x)
        out=self.classify(x)
        return out


if use_cuda:
    model=TextCNNBN_model(len_dic,emb_dim,input_dim).cuda()
else:
    model=TextCNNBN_model(len_dic,emb_dim,input_dim)

criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=learning_rate)
best_acc=0
best_model=None
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, data in enumerate(trainDataLoader):
        x, y = data
        if use_cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()
        else:
            x, y = Variable(x), Variable(y)
        # forward
        out = model(x)
        loss = criterion(out, y)
        train_loss += loss.data * len(y)
        _, pre = torch.max(out, 1)
        num_acc = (pre == y).sum().item()
        train_acc += num_acc
        # backward
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()
        if (i + 1) % 100 == 0:
            print('[{}/{}],train loss is:{:.6f},train acc is:{:.6f}'.format(i, len(trainDataLoader),
                                                                            train_loss / (i * batch_size),
                                                                            train_acc / (i * batch_size)))
    print(
        'epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                     train_loss / (len(trainDataLoader) * batch_size),
                                                                     train_acc / (len(trainDataLoader) * batch_size)))
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for i, data in enumerate(testDataLoader):
        x, y = data
        if use_cuda:
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.data * len(y)
        _, pre = torch.max(out, 1)
        num_acc = (pre == y).sum().item()
        eval_acc += num_acc
    print('test loss is:{:.6f},test acc is:{:.6f}'.format(
        eval_loss / (len(testDataLoader) * batch_size),
        eval_acc / (len(testDataLoader) * batch_size)))
    if best_acc<(eval_acc / (len(testDataLoader) * batch_size)):
        best_acc=eval_acc / (len(testDataLoader) * batch_size)
        best_model=model.state_dict()
        # print(best_model)
        print('best acc is {:.6f},best model is changed'.format(best_acc))

torch.save(model.state_dict(),'./model/TextCNNBN_model.pth')