from PIL import Image
from torchvision import transforms
"""
该部分代码主要是学习使用pytorch中图形转换模块transforms的过程，其中包括了简单的类的使用
如：totensor（将图片转化为tensor数据类型）；normalize（将tensor数据进行归一化）；compose（将不同的transforms组合起来）
"""
img_path = "C:/Users/69129/Pictures/Screenshots/pic.png"
img = Image.open(img_path)
"""
tensor_tool = transforms.ToTensor()
img_tensor = tensor_tool(img)
normalize_tool = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_normal = normalize_tool(img_tensor)
"""

transform = (transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
]))
img_Tensor = transform(img)
print(img_Tensor)