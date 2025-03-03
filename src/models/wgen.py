import torch.nn as nn
import torch.nn.functional as F
import torch


class wgen(nn.Module):
    def __init__(self, z_dim = 128, output_dim = 128):
        super(wgen, self).__init__()
        self.main = nn.Sequential(
            # 输入是一个z_dim维的噪声向量
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            # 输出是一个10维的logits向量
            nn.Linear(1024, output_dim),
            # 可以使用一个线性层或者一个softmax层
            # nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.main(input)
        return output


if __name__ == '__main__':    
    z = torch.randn(10)
    print(z)
    model = wgen(z_dim = 10, output_dim= 10)
    output = model(z)
    print(output)