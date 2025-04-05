# 该模块主要是为了实现Unet里面的横向卷积过程，每个上下采样后，都需要进行两次卷积，主要是改变图像通道数，和容易batch_t_emding信息

'''
# Part1 引入相关的库函数
'''
import torch
from dataset import *
from torch import nn

'''
# Part2 将卷积设置为一个类函数 
'''


class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, time_emding_size):
        super().__init__()

        # 首先要对图像进行第一个卷积，改变通道，不改变大小
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # 第一个卷积后需要融入t_embdding信息，然后再第二次卷积,所以这里需要把维度先统一下,统一通道，然后加到对应图像上
        self.linear1 = nn.Linear(time_emding_size, out_channel)
        self.relu = nn.ReLU()

        # 第二个卷积,不改变通道，和大小
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, batch_xt, batch_t_embdding):  # (batch,channel,imag_size,imag_size) # (batch,emdding_size)
        # 先对图像进行特征提取
        x = self.conv1(batch_xt)  # (batch,out_channel,imag_size,imag_size)
        # 在对时间t进行维度的统一和扩展。
        t = self.linear1(batch_t_embdding)
        t = t.reshape(t.size()[0], t.size()[1], 1, 1)  # (batch,ou_channel,1,1)
        t = self.relu(t)
        # 融合噪声图和t，并进行第二次卷积
        output = self.conv2(x + t)
        return output
