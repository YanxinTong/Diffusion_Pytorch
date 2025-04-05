# 该模块是为了实现对于t的编码，从而可以和加噪声的图像一起进行输入Unet里面，输出t-1时刻的噪声·
# 由于该模块是输入部分，并且需要训练，是模型的一部分，所以需要作为模型的类里面
'''
# 首先第一步，是引入相关的库函数
'''

import torch
from config import *
from torch import nn

'''
# 第二步定义一下位置编码类，初始化需要输入emding的维度，运行的时候，只需要输入batch_t就行，输出batch_emding
'''
class TimePositionEmbedding(nn.Module):
    def __init__(self,emding_size):
        super().__init__()
        seq_len=T
        seq_index=torch.arange(0,seq_len).unsqueeze(1) # (seq_len,1)
        emd_fill=seq_index*torch.exp(-torch.log(torch.tensor(10000))*torch.arange(0,emding_size,2)/emding_size)
        # (seq_len,1)*(emd_size//2,1)=(seq_len,emd_size//2)

        time_position_emb=torch.zeros(size=(seq_len,emding_size))
        # 偶数sin
        time_position_emb[:,0::2]=torch.sin(emd_fill)
        # 奇数cos
        time_position_emb[:,1::2]=torch.cos(emd_fill)
        # 保存数据
        self.register_buffer('time_position_emb',time_position_emb)

    def forward(self,batch_t): 
        return self.time_position_emb[batch_t]


if __name__ == '__main__':
    time_pos_emb = TimePositionEmbedding(8)
    t = torch.randint(0, T, (2,))  # 随机2个图片的time时刻
    embs_t = time_pos_emb(t)
    print(embs_t)