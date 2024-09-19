import torch

"""
我在这个代码里进行了关于tensor这个数据类型在python和pytorch中的学习， 其中包括如何使用张量这样的数据类型，如何使用不同维度的张量的数据，张量的索引和切片等操作
以及一些pytorch中的处理张量的函数，如生成0张量，生成全是随机数的张量，张量的升维和降维
"""
#张量加法和生成0张量
Tensor = torch.zeros(3, 3, 3)   #只能生成三阶的0张量，不能把张量归0
print(Tensor)
print("----------------------------------------")
tensor = Tensor[:, 2, :]
print(tensor)
print("----------------------------------------")
Tensor += Tensor
print(Tensor)
print("----------------------------------------")

#张量的升维，降维操作以及对不同维度进行操作（切片，索引）
Tensor = torch.randn(3, 2, 1)
print(Tensor)
print("----------------------------------------")
Tensor = Tensor.squeeze(2)
print(Tensor)
print("----------------------------------------")
Tensor = Tensor.unsqueeze(1)    #Tensor[3 ,1, 2]
print(Tensor)
print("----------------------------------------")
Tensor = Tensor.squeeze(0)
print(Tensor)   #根据结果来看由于这个时候维度没有降下去因此推测squeeze降维函数应该是只会降下理论上可以降下的维度
print("----------------------------------------")
tensor = Tensor
Tensor = Tensor[:, :, 0:1]
print(Tensor)
print("----------------------------------------")
tensor = tensor[:, :, :-1]      #列表元素的倒序,此时tensor为[3, 1, 1]
print(tensor)
print("----------------------------------------")
tensor = tensor.unsqueeze(2)    #升维后tensor为[3, 1, 1, 1]
print(tensor)
print("----------------------------------------")
#张量的维度的改造（-1为计算机自己判断，维持总的元素个数不变原则）
matrix = torch.randn(3, 4, 5)
print(matrix)
print("----------------------------------------")
matrix = matrix.view(-1, 3, 4, 5)
print(matrix)
#张量的连接以及arange的功能
ten = torch.arange(0, 10)
print(ten)
ten = torch.arange(0, 10, 2)
print(ten)
t = torch.cat([ten, ten], dim = 0)
A = torch.randn(3, 4, 5)
B = torch.randn(3, 4, 5)
print(A)
print("----------------------------------------")
print(B)
print(torch.cat([A, B], dim = -1))
print("----------------------------------------")
print(t)
m = torch.arange(0, 5)
print(m)
#爱因斯坦求和以及sin和cos作用于张量的结果
re = torch.einsum('i ,j -> ij', m, t)
print(re)
sin = re.sin()
print(sin)
print("----------------------------------------")
#矩阵的相乘及转置
a = torch.randn(3, 4)
b = torch.randn(3, 4)
print(a)
print("----------------------------------------")
print(b)
print("----------------------------------------")
print(b.T)
c = a @ b.T
print(c)
#张量维度的变化
temp = torch.randn(1, 2, 3, 1)
print(temp)
print("----------------------------------------")
temp = temp.transpose(1, 2)
print(temp)
