import torch
from torch import nn
import torch.nn.functional as func
"""
RNN的本质实际上是线性变换，将循环结构拆分后可发现RNN实际上是由线性层组成的：
一个是将原始数据输入隐藏层；另一个是将上一个隐藏层的数据输入到该隐藏层
结合到RNN模型的理论部分以及pytorch中的模块可以进行简单的RNN模型的构建
题目中所要求的nn.Linear为pytorch中的实现线性层的模块，因此可以利用其来构建RNN模型
"""
class New_hidden(nn.Module):#对数据进行处理，得到新的隐藏层
    def __init__(self, input_size, hidden_size):
        super(New_hidden, self).__init__()
        self.input_size = input_size#初始化输入层的大小
        self.hidden_size = hidden_size#初始化隐藏层的大小
        self.i_to_h = nn.Linear(input_size, hidden_size)#定义从输入层到隐藏层的线性变换
        self.h_to_h = nn.Linear(hidden_size, hidden_size)#定义从隐藏层到隐藏层的线性变换

    def forward(self, x, hidden):
        combined = self.i_to_h(x) + self.h_to_h(hidden)
        new_hidden = func.tanh(combined)
        return new_hidden

class MyRNN(nn.Module):#搭建RNN的模型
    def __init__(self, input_size, hidden_size):#实际上输出层的维度与隐藏层的维度相同，故利用隐藏层的维度来代替输出层的维度
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size#初始化隐藏层的大小
        self.h_to_o = nn.Linear(hidden_size, hidden_size)#定义从隐藏层到输出层的线性变换
        self.new_hidden = New_hidden(input_size, hidden_size)

    def forward(self, x):#定义RNN的前进层
        batch_size, seq_len, _= x.size()#获取样本的批次的数据和序列长度
        hidden = torch.zeros(batch_size, self.hidden_size)
        for data in range(seq_len):#对序列进行循环的处理
            hidden = self.new_hidden(x[:, data,: ], hidden)
        output = self.h_to_o(hidden)#获取最终的输出层
        return output

inputsize = 10
hiddensize = 20
mymodule = MyRNN(inputsize, hiddensize)
data = torch.randn(5, 7, inputsize)
output = mymodule(data)
print(output)


