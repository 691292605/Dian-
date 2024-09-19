import torch
import torch.nn as nn
from torch import optim

"""
实现多头自注意力的过程中可以参考实现自注意力的过程，与自注意力的实现相比，在这个过程中多了一步将q,k,v矩阵进行“分头”然后在将
分开的q,k,v合并起来的操作，在这个过程中尤其是需要注意张量维度的变化，同时这里为了减少转置操作的次数，在将q,k,v升维的时候选择顺便将其重构为
不同的结构。
"""

class Multi_Head_attention(nn.Module):
    def __init__(self, input_size, hidden_size, head):
        super(Multi_Head_attention, self).__init__()
        self.head = head
        self.hidden_size = hidden_size
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, data):
        seq_len = data.size(1)
        batch_size = data.size(0)
        q = self.Wq(data)
        k = self.Wk(data)
        v = self.Wv(data)
        q = q.view(batch_size, self.head, seq_len, self.hidden_size // self.head)
        k = k.view(batch_size, self.head, self.hidden_size // self.head, seq_len)
        v = v.view(batch_size, self.head, self.hidden_size // self.head, seq_len)
        weight = torch.softmax(q @ k, dim = -1)
        b = v @ weight
        b = b.view(batch_size, -1, seq_len).transpose(1, 2)
        out = self.out(b)
        return out


#示例
#注意，隐藏层维度需要是head数的整数倍
Batch_size = 5
Hidden_size = 10
Input_size = 5
seq_size = 6
Head = 2
my_module = Multi_Head_attention(Input_size, Hidden_size, Head)
Data = torch.randn(Batch_size, seq_size, Input_size)
target = torch.randn(Batch_size, seq_size, Input_size)
result = my_module(Data)
print(result)

loss = nn.MSELoss()
optimizer = optim.Adam(my_module.parameters(), lr = 0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = my_module(Data)
    Loss = loss(output, target)
    Loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("loss:{}".format(Loss))
