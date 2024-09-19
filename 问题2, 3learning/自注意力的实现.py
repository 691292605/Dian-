import torch
import torch.nn as nn
from torch import optim


class attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(attention, self).__init__()
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)

    def forward(self, data):
        q = self.Wq(data)
        k = self.Wk(data)
        v = self.Wv(data)
        dim = q.size(-1)
        weights = torch.softmax(k.T @ q / dim ** 0.5, dim = -1)
        attention_weight = v @ weights
        return attention_weight, weights



Input_size = 5
Hidden_size = 10
seq_len = 2

x = torch.randn(seq_len, Input_size)
target = torch.randn(seq_len, Hidden_size)
my_attention = attention(Input_size, Hidden_size)
print("x: ", x)
print("target: ", target)

loss = nn.MSELoss()
#不是分类任务不用交叉熵了
optimizer = optim.Adam(my_attention.parameters(), lr = 0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output, weight = my_attention(x)
    Loss = loss(output, target)
    Loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("loss:{}".format(Loss))
        print("weight: ", weight)
