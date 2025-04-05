# 该模块实现的是引用一个MINIST训练数据集，需要预处理

'''
# Part1 引入一些库函数
'''

import torch
from torch.utils import data
import torchvision
from torchvision import transforms
from config import *

# 显示图像
import matplotlib.pyplot as plt

PiltoTensor_action=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), # 改变图像的大小
    transforms.ToTensor()] # 对图像转化为Tensor类型，归一化，并且把通道提前(H,W,C)->(C,H,W)
)

minist_train=torchvision.datasets.MNIST(root='dataset',train=True,transform=PiltoTensor_action,download=True)

minist_loader=data.DataLoader(dataset=minist_train,batch_size=2,shuffle=True) # (image,label)


# 为了可以简单的按照pillow去展示自己的图像，那么需要一个把tensor转回为Pillow的操作。主要分为三步。
TenosrtoPil_action=transforms.Compose([
    transforms.Lambda(lambda t:t*255),
    transforms.Lambda(lambda t:t.type(torch.uint8)),
    transforms.ToPILImage()
])

if __name__ == '__main__':
    Tensor_imag=minist_train[0][0]
    print(Tensor_imag)
    plt.figure(figsize=(5,5)) # 绘制画布
    pil_imag=TenosrtoPil_action(Tensor_imag)
    plt.imshow(pil_imag) # 绘制图像，和plot，bar等函数一样用于绘制不同的图像
    plt.show() # 展示图像