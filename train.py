# 该模块主要是为了实现Diffusion模型的训练。

'''
# Part1 引入相关的库函数
'''

import torch
from dataset import minist_loader
from torch import nn
from config import *
# 相关的模型的引入
from unet import Unet
from diffusion import forward_diffusion

'''
# Part2 定义一些初始化模型, 优化器等等。
'''

net = Unet(imag_channel=1)
loss_f = nn.L1Loss()

lr = 1e-3

opt = torch.optim.Adam(params=net.parameters(), lr=lr)

# Part3 开始训练
if __name__ == '__main__':

    # 定义一些训练参数。
    epoch = 1000

    # 开始训练
    net.train()
    n_iter = 0
    for i in range(epoch):
        for imags, labels in minist_loader:
            # 首先统一数值范围
            imags = imags * 2 - 1  # 将范围从(0,1) -> (-1,1)
            # 第一步随机生成batch_t
            batch_t = torch.randint(low=0, high=T, size=(imags.size()[0],))
            # 生成对应的x_t和对应真值噪音
            batch_x_t, batch_noise_t = forward_diffusion(imags, batch_t)
            # 将两者输入Unet进行输出
            batch_noise_pre = net(batch_x_t, batch_t)  # 得到噪声(batch,channel,imag_size,imag_size)
            # 计算损失
            l = loss_f(batch_noise_pre, batch_noise_t)
            # 清除梯度
            opt.zero_grad()
            # 反向传播得到梯度
            l.backward()
            # 更新参数
            opt.step()
            # 累加损失
            loss1 = l.item()
            # 更新迭代次数
            n_iter+=1

            print('当前的iter为{},当前损失为{}'.format(n_iter, loss1))
        torch.save(net, f='unet_epoch{}.pt'.format(i))
