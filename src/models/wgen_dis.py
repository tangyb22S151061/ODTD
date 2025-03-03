import torch
from torch import nn


class wgen_dis(torch.nn.Module):
    def __init__(self, input_dim=256):
        super(wgen_dis, self).__init__()
        self.main = nn.Sequential(
            # 输入是一个 input_dim 维的logits向量
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出是一个标量，表示真实性得分
            nn.Linear(256, 1),
            # 可以使用一个线性层或者一个sigmoid层
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
