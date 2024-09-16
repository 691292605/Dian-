import torch
from torch.nn import L1Loss
"""
该部分代码主要为学习使用损失函数的过程，其中包含了reshape函数的使用
以及一些简单使用损失函数计算误差的示例
"""

Input = torch.tensor([1.0, 2.0, 3.0])
Target = torch.tensor([1.0, 2.0, 5.0])

inputs = torch.reshape(Input, [1, 1, 1, 3])
targets = torch.reshape(Target, [1, 1, 1, 3])

loss = L1Loss()
result = loss(inputs, targets)
print(result)

x = torch.tensor([0.1, 0.2, 0.3, 0.4])
y = torch.tensor([1])
print(x)
x = torch.reshape(x, (1, 4))
print(x)
Loss = torch.nn.CrossEntropyLoss()
Result = Loss(x, y)
print(Result)