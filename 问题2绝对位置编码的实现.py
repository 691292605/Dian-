import torch
import torch.nn as nn
import math


"""
实现绝对编码较为简单，理解它是作为位置编码直接与词向量矩阵相加即可在词向量上添加位置信息即可
在这个过程中只需要实现绝对编码编码值的计算并将其加到词向量矩阵上即可
"""

class Encoding(nn.Module):
    def __init__(self, max_len, dim):
        super(Encoding, self).__init__()
        if dim % 2 != 0:    #在测试时发现，该绝对编码过程无法处理维度为奇数的情况，故这里增加了对奇数的处理情况
            dim += 1
            self.is_odd = True
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.arange(0, dim, 2)
        div_term = torch.exp(div_term * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if self.is_odd:
            pe = pe[:, :-1]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, data):
        seq_len = data.size(1)
        return data + self.pe[:, :seq_len]


#举例
Len = 10
d = 5
model = Encoding(Len, d)
x = torch.randn(2, 5, d)
output = model(x)
print("x:")
print(x)
print("pe:")
print(model.pe)
print("output:")
print(output)
