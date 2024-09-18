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
        if dim % 2 != 0:    #在代码编写完之后测试发现，该绝对编码过程无法处理维度为奇数的情况，故这里增加了对奇数的处理情况
            dim += 1
            self.is_odd = True
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        func = torch.arange(0, dim, 2)
        func = torch.exp(func * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * func)
        pe[:, 1::2] = torch.cos(position * func)
        #具体公式的推导和运行的过程在note里
        """
        直接在这里进行位置编码的计算是因为每个位置的位置编码都是固定的，与参数无关，只与位置有关
        也就是说在init中直接计算可以避免在下面的函数中进行重复计算
        """
        if self.is_odd:
            pe = pe[:, :-1]
        self.register_buffer('pe', pe.unsqueeze(0))     #为了使计算的编码保存在模型本身中，保证不会随着下面的计算而改变

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
