import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


tensor_tool = transforms.ToTensor()
transform = (transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
train_dataset = FashionMNIST(root = './data', train = True, transform = transform, download = True)
test_dataset = FashionMNIST(root = './data', train = False, transform = transform, download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 1000, shuffle = False)


class New_hidden(nn.Module):    #对数据进行处理，得到新的隐藏层
    def __init__(self, input_size, hidden_size):
        super(New_hidden, self).__init__()
        self.input_size = input_size    #初始化输入层的大小
        self.hidden_size = hidden_size      #初始化隐藏层的大小
        self.i_to_h = nn.Linear(input_size, hidden_size)    #定义从输入层到隐藏层的线性变换
        self.h_to_h = nn.Linear(hidden_size, hidden_size)   #定义从隐藏层到隐藏层的线性变换

    def forward(self, x, hidden):
        combined = self.i_to_h(x) + self.h_to_h(hidden)
        new_hidden = torch.tanh(combined)
        return new_hidden


class MyRNN(nn.Module):     #搭建RNN的模型
    def __init__(self, input_size, hidden_size, output_size):    #实际上输出层的维度与隐藏层的维度相同，故利用隐藏层的维度来代替输出层的维度
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size  #初始化隐藏层的大小
        self.h_to_o = nn.Linear(hidden_size, output_size)   #定义从隐藏层到输出层的线性变换
        self.new_hidden = New_hidden(input_size, hidden_size)

    def forward(self, x):   #定义RNN的前进层
        batch_size, seq_len, _= x.size()    #获取样本的批次的数据和序列长度
        hidden = torch.zeros(batch_size, self.hidden_size)
        for data in range(seq_len):     #对序列进行循环的处理
            hidden = self.new_hidden(x[:, data,: ], hidden)
        output = self.h_to_o(hidden)    #获取最终的输出层
        return output


inputsize = 28     #Fashion Mnist的数据集是28*28的图像
hiddensize = 100
outputsize = 10     #输出共有10个种类

mymodel = MyRNN(inputsize, hiddensize, outputsize)     #定义模型

criterion = CrossEntropyLoss()    #定义损失函数，对于分类问题选择交叉熵
optimizer = optim.Adam(mymodel.parameters(), lr=0.001)      #定义优化器
total_train_step = 0
total_test_step = 0
train_size = len(train_dataset)
test_size = len(test_dataset)
print("训练集的长度为:{}".format(train_size))
print("测试集的长度为:{}".format(test_size))

epochs = 5
for epoch in range(epochs):
    print("------------第{}轮训练开始---------------".format(epoch + 1))
    #mymodel.train()        实际对模型没有什么作用
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.view(-1, 28, 28)
        outputs = mymodel(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        total_train_step = total_train_step + 1
        if total_train_step % 500 == 0:
            print("训练总次数:{}，本次训练中的损失={}".format(total_train_step, loss.item()))
    print("本轮训练的总损失={}".format(running_loss))
#开始测试:
    #mymodel.eval()     实际对模型没有什么作用
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for Data in test_loader:
            imgs, targets = Data
            imgs = imgs.view(imgs.size(0), 28, 28)
            outputs = mymodel(imgs)
            result_loss = criterion(outputs, targets)
            total_test_loss = total_test_loss + result_loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整个测试集上的损失={}".format(total_test_loss))
    print("整体测试集的正确率={}".format(total_accuracy / test_size))

#在该模型中以测试集中分类的正确率作为分类的指标评判模型的效果


