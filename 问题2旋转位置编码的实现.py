import torch
import torch.nn as nn

"""
在实现旋转位置编码的时候选择一步步得到最后进行元素级相乘的矩阵
在这个代码中，x的数据类型为：x[seq_len, dim]。但是由于旋转位置编码的两两进行旋转操作的特性以及时间问题，对于维度为奇数的情况这里不再进行优化，
只是实现了简单的维度为偶数的旋转位置编码
另外考虑到问题3的存在，这里的旋转位置编码没有涉及到自注意力的部分（进行计算的的不是q和k），只是进行了编码的实现

两种编码的区别：
结合问题二两种位置编码的实现过程中，我发现了绝对位置编码是每个位置编码是固定的，比较“绝对”， 而相对位置编码由于有一步两两旋转的操作，
因此旋转位置编码则是刻画了一个相对的位置，其对于位置信息的处理更为灵活，不像绝对位置编码具有局限性，但是旋转位置编码需要更多的计算
"""

class rope(nn.Module):
    def __init__(self, dim):
        super(rope, self).__init__()
        self.dim = dim

    def forward(self, x):
        seq_len = x.size(0)
        theta = 1 / (10000 ** (torch.arange(0, self.dim, 2) // self.dim)).float()
        cat_theta = torch.cat([theta, theta], dim=0)
        m = torch.arange(0, seq_len)
        m_theta = torch.einsum('i ,j -> ij', m, cat_theta)
        sin = m_theta.sin()
        cos = m_theta.cos()
        x_front = x[:, : self.dim // 2]
        x_back = x[:, self.dim // 2:]
        x_pass = torch.cat([-x_back, x_front], dim = -1)
        output = (x * cos + x_pass * sin)
        return output

d = 8
data = torch.randn(3, d)
Rope = rope(d)
result = Rope(data) + data
print("data: ", data, sep = '\n')
print("rope：", Rope(data), sep = '\n')
print("result: ", result, sep = '\n')
