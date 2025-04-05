# 该模块实现的过程是利用函数输入批次的图像和批次的t，对图像利用公式进行正向扩散，得到批次的噪声图，
# 注意，这里实现的时候不是类，因为这个严格来说不算网络结构，也就是不在计算图的构建范围内，只是构造训练数据集的其中一个处理过程，所以写成函数即可。

'''
# Part1 引入相关的库函数
'''
from config import * # 一些基础参数
from dataset import minist_train, minist_loader,TenosrtoPil_action # 数据的测试
import torch # 用一系列的数据处理
import matplotlib.pyplot as plt # 用于绘图

'''
# Part2 初始化一些beta和alpha参数用于函数中的加噪过程。
'''

# 首先获取Beta_t,主要是T步里面的参数，这个一般是直接定好的，(left，right)中取样T次(单调增加的)
beta_t=torch.linspace(start=0.0001,end=0.02,steps=T) # (T,) # 单维度tensor

# 然后是获取alpha_t ，因为和beta_t相加为1，所以直接和1相减 (单调递减)
alpha_t=1-beta_t # (T,)

# 然后需要得到alpha_bar,这个是累乘得到的
alpha_bar=torch.cumprod(alpha_t,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]

# 定义函数 forward_diffusion，输入批次图像和批次t，然后输出对应的批次的x_t，和对应的t-1时刻加入的噪声图。
def forward_diffusion(batch_x0,batch_t): # (batch_size,chanal,imag_sie,imag_size) , (batch_size,)
    # 第一步首先要获取，整个batch图像的，t-1时刻的噪声。
    batch_noise_t=torch.randn_like(input=batch_x0) # (batch_size,chanal,imag_sie,imag_size),默认是标准正态分布
    # 首先需要利用batch_t,从alpha_bar里面取出对应的t,并且为了便于广播机制，需要对batch_t进行形状的转换，至少保持同纬度
    alpha_bar_t=alpha_bar[batch_t].reshape(batch_t.size()[0],1,1,1)
    # 计算得到噪声后的图
    batch_xt=torch.sqrt(alpha_bar_t)*batch_x0+torch.sqrt(1-alpha_bar_t)*batch_noise_t
    return batch_xt,batch_noise_t


if __name__ == '__main__':
    batch_x = next(iter(minist_loader))[0]  # 2个图片拼batch, (2,1,48,48)

    # 加噪前的样子
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(TenosrtoPil_action(batch_x[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(TenosrtoPil_action(batch_x[1]))
    plt.show()
    # 如果需要将噪声（通常是从[-1, 1]范围生成的）加到图像上，你需要将图像数据重新缩放到[-1, 1]范围，以便它与噪声匹配，能够平衡噪声对训练的影响。
    # 虽然正态分布不是严格[-1,1]之间，但是通过三sigema定理，我们这里可以初略定在一个sigema之间。
    # 总之就是没有偏差
    batch_x = batch_x * 2 - 1  # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_t = torch.randint(0, T, size=(batch_x.size(0),))  # 每张图片随机生成diffusion步数
    # batch_t=torch.tensor([5,100],dtype=torch.long)
    print('batch_t:', batch_t)

    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())

    # 加噪后的样子
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(TenosrtoPil_action((batch_x_t[0] + 1) / 2)) # 返回原来的图像范围
    plt.subplot(1, 2, 2)
    plt.imshow(TenosrtoPil_action((batch_x_t[1] + 1) / 2))
    plt.show()







