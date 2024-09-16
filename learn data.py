from torch.utils.data import DataLoader
from torchvision import transforms, datasets
"""
该部分代码主要为使用torchvision中的data的学习过程，其中包括如何得到pytorch中的数据（搭建dataset）
如何将数据集中的数据传入循环网络（搭建dataloader）
如何调取dataloader中的返回值
"""

tensor_tool = transforms.ToTensor()
normal_tool = transforms.Normalize(0.5, 0.5)
transform = transforms.Compose([tensor_tool, normal_tool])
#搭建dataset和dataloader
train_dataset = datasets.FashionMNIST(root = './data', train = True,transform = transform, download = False )
test_dataset = datasets.FashionMNIST(root = './data', train = False,transform = transform, download = False )
test_loader = DataLoader(dataset = test_dataset, batch_size = 10, shuffle = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = 10, shuffle = True)
#调取dataset中的返回值
img, target = train_dataset[0]
print(img.shape)
print(target)
#调取dataloader中的返回值
epochs = 2
for epoch in range(epochs):#由于对数据进行了打乱（shuffle = True)，故再次进行
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)