# 该模块主要为了实现Unet模块，输入为emdding的t(batch_size,emdding_size)和
# 加了t次噪声的x_t(batch_size,channel,imag_size,imag_size)，输出为去噪后的图像(batch_size,channel,imag_size,imag_size)

'''
# Part1 引入相关的库函数
'''
import torch
from config import *
from torch import nn
from conv_block import Conv_Block
from time_position_emb import TimePositionEmbedding
from dataset import minist_loader
from diffusion import forward_diffusion

'''
# 定义Unet的类函数，来实现预测t-1时刻的噪声图
'''

class Unet(nn.Module):
    def __init__(self,imag_channel,emd_size=256,channels=[64,128,256,512,1024]):
        super().__init__()
        # 开始初始化各个参数，包含卷积和各种采样。
        # 首先对t进行嵌入和线性变化(虽然我不太清楚为什么要线性变化，纯为了提取信息？)
        self.time_position_emb=nn.Sequential(
            TimePositionEmbedding(emd_size),
            nn.Linear(emd_size,emd_size),
            nn.ReLU()
        )

        # 第二部对于encoder部分，需要进行卷积和(改通道)，以及下采样。
        channels=[imag_channel]+channels
        self.emd_size=emd_size


        # 初始化通道变化的编码器卷积
        self.encoder_conv=nn.ModuleList(
            [Conv_Block(in_channel=channels[i],out_channel=channels[i+1],time_emding_size=emd_size) for i in range(len(channels)-1)]
        )

        # 初始化下采样的max_polling,核的边长等于步长的时候，相当于缩小边长平方被的面积(当然前提是图像边长为核边长的倍数)，所以这里相当于边长缩小一半
        self.encoder_max=nn.ModuleList(
            [nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0) for _ in range(len(channels)-2)]
        )

        # 第三部分：开始解码器的卷积，和解码器的上采样，注意解码过程需要少一个卷积和少一个上采样，然后添加上一个output卷积用于统一通道数

        # 初始化通道编码的解码器的卷积，通道和原来反着变化，但是要少一个
        self.decoder_conv=nn.ModuleList(
            [Conv_Block(in_channel=channels[-i-1],out_channel=channels[-i-2],time_emding_size=emd_size) for i in range(len(channels)-2)]
        )
        # 初始化上采样的解码器，通道也和原来反着来，并且少一个。(注意为什么，下采样不变通道数，但是这里要变通道数，因为上采样多了个残差，所以通道有变2倍)
        self.decoder_up=nn.ModuleList(
            [nn.ConvTranspose2d(in_channels=channels[-i-1],out_channels=channels[-i-2],kernel_size=(2,2),stride=2,padding=0) for i in range(len(channels)-2)]
        )
        # 变化通道为初始的大小
        self.output_conv=nn.Conv2d(in_channels=channels[1],out_channels=imag_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,batch_x_t,batch_t):
        batch_t_emd=self.time_position_emb(batch_t) # (batch_size,emdding_size)

        # 开始下采样,并且存储每次的conv结果，作为下次的残差部分
        resdual=[]
        encoder_output=batch_x_t
        for i in range(len(self.encoder_max)):
            encoder_output=self.encoder_conv[i](encoder_output,batch_t_emd)
            resdual.append(encoder_output)
            encoder_output=self.encoder_max[i](encoder_output)

        # encoder的时候，卷积个数是比下采样多一个的，所以最后一个需要额外进行卷积，并且不需要记录
        encoder_output=self.encoder_conv[-1](encoder_output,batch_t_emd)


        # 此时得到了resdual的结果，和最终的encoder结果，开始解码
        decoder_output=encoder_output
        for i in range(len(self.decoder_up)):
            # 先上采样
            decoder_output=self.decoder_up[i](decoder_output)
            # 进行残差，通道维度进行合并
            decoder_output=torch.cat((resdual[-i-1],decoder_output),dim=1) #
            # 再进行反卷积
            decoder_output=self.decoder_conv[i](decoder_output,batch_t_emd)

        # 最后进行通道的返回
        return self.output_conv(decoder_output) # (batch,imag_channel,imag_size,imag_size) # 这个就是噪声

if __name__=='__main__':
    batch_x=next(iter(minist_loader))[0] # 2个图片拼batch, (2,1,48,48)
    batch_x=batch_x*2-1 # 像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配

    batch_t=torch.randint(0,T,size=(batch_x.size(0),))  # 每张图片随机生成diffusion步数
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)

    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    unet=Unet(imag_channel=1)
    batch_predict_noise_t=unet(batch_x_t,batch_t)
    print('batch_predict_noise_t:',batch_predict_noise_t.size())







