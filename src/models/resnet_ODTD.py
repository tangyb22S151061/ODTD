"""
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""

import torch
import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_mnist(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet_mnist, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        x = self.fc(features)

        return x

    def forward2(self, x):
        # 与forward方法相同的前向传播过程，直到avgpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # 将x展平为一维向量，得到logits
        features = x.view(x.size(0), -1)
        # 不再经过fc层，直接返回logits
        return features
    
    def getFeaturesSize(self):
        return 64
    

class ResNet_mnist_defense(nn.Module):
    def __init__(self, block, layers, num_classes=10, discriminator=None, T_ft=None, rank_map=None, defense=False):
        super(ResNet_mnist_defense, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.discriminator = discriminator  # 判别器
        self.T_ft = T_ft  # 微调模型
        self.rank_map = rank_map  # 等级图
        self.defense = defense

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def generate_noise(x):
        # 获取模型的预测
        output = x
        # 获取top-1预测
        _, top1_pred = output.topk(1, dim=1)

        noise = None
        while True:
            # 生成一个随机的logits张量
            noise = torch.rand_like(output)
            # 获取噪声的top-1预测
            _, noise_top1_pred = noise.topk(1, dim=1)
            # 如果噪声的top-1预测与模型的top-1预测不同，就跳出循环
            if noise_top1_pred != top1_pred:
                break

        return noise

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)

    #     x = self.avgpool(x)
    #     features = x.view(x.size(0), -1)
    #     x = self.fc(features)
    #     return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # features
        x = x.view(x.size(0), -1)
        
        # 如果添加了还需要额外的预处理
        if self.defense:
            # 判断预测值是否在等级图范围内
            min_rank = min(self.rank_map)
            max_rank = max(self.rank_map)
            if not min_rank <= x <= max_rank:
                # 输出top-1和T.pt不相等的随机噪声
                x = self.generate_noise(x)
            else:
                # 比较判别器值比多少等级范围的值低
                lower_ranks = sum([1 for rank in self.rank_map if self.discriminator(self.T_ft.forward2(x)) < rank])
                # 衰减温度为0.9的多少次方
                temperature = 0.9 ** lower_ranks
                x = x / temperature
            
        x = self.fc(x)
        return x
    def getFeaturesSize(self):
        return 64



class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """定义BasicBlock残差块类
        
        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, ksize=3, num_classes=100, in_channels = 3, zero_init_residual=False):
        
        """
        定义ResNet网络的结构
        
        参数：
            block (BasicBlock / Bottleneck): 残差块类型
            layers (list): 每一个stage的残差块的数目，长度为4
            num_classes (int): 类别数目
            zero_init_residual (bool): 若为True则将每个残差块的最后一个BN层初始化为零，
                这样残差分支从零开始每一个残差分支，每一个残差块表现的就像一个恒等映射，根据
                https://arxiv.org/abs/1706.02677这可以将模型的性能提升0.2~0.3%
        """

        super(ResNet, self).__init__()
        self.inplanes = 64  # 第一个残差块的输入通道数
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=ksize, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        定义:
            ResNet的一个Stage的结构
        
        参数：
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            bloacks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def getFeaturesSize(self):
        return 512

# ResNet18 for MNIST
class ResNet_defense(nn.Module):

    def __init__(self, block, layers, ksize=3, num_classes=100, in_channels = 3, zero_init_residual=False, discriminator=None, T_ft=None, rank_map=None, defense=False):
        
        """
        定义ResNet网络的结构
        
        参数：
            block (BasicBlock / Bottleneck): 残差块类型
            layers (list): 每一个stage的残差块的数目，长度为4
            num_classes (int): 类别数目
            zero_init_residual (bool): 若为True则将每个残差块的最后一个BN层初始化为零，
                这样残差分支从零开始每一个残差分支，每一个残差块表现的就像一个恒等映射，根据
                https://arxiv.org/abs/1706.02677这可以将模型的性能提升0.2~0.3%
        """

        super(ResNet_defense, self).__init__()
        self.inplanes = 64  # 第一个残差块的输入通道数
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=ksize, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage1 ~ Stage4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.discriminator = discriminator  # 判别器
        self.T_ft = T_ft  # 微调模型
        self.rank_map = rank_map  # 等级图
        self.defense = defense

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        定义:
            ResNet的一个Stage的结构
        
        参数：
            block (BasicBlock / Bottleneck): 残差块结构
            plane (int): 残差块中第一个卷积层的输出通道数
            bloacks (int): 当前Stage中的残差块的数目
            stride (int): 残差块中第一个卷积层的步长
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def generate_noise(x):
        # 获取模型的预测
        output = x
        # 获取top-1预测
        _, top1_pred = output.topk(1, dim=1)

        noise = None
        while True:
            # 生成一个随机的logits张量
            noise = torch.rand_like(output)
            # 获取噪声的top-1预测
            _, noise_top1_pred = noise.topk(1, dim=1)
            # 如果噪声的top-1预测与模型的top-1预测不同，就跳出循环
            if noise_top1_pred != top1_pred:
                break

    def forward(self, x):
        pred_x = self.T_ft.forward2(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # features
        x = x.view(x.size(0), -1)
        if self.defense:
            # 判断预测值是否在等级图范围内
            min_rank = min(self.rank_map)
            max_rank = max(self.rank_map)
            pred = self.discriminator(pred_x)
            if not min_rank <= pred <= max_rank:
                # 输出top-1和T.pt不相等的随机噪声
                x = self.generate_noise(x)
            else:
                # 比较判别器值比多少等级范围的值低
                lower_ranks = sum([1 for rank in self.rank_map if self.discriminator(self.T_ft.forward2(x)) < rank])
                # 衰减温度为0.9的多少次方
                temperature = 0.9 ** lower_ranks
                x = x / temperature
        x = self.fc(x)
        return x
    
    
    def getFeaturesSize(self):
        return 512



class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        logits = x.view(x.size(0), -1)
        x = self.fc(logits)

        return x
    
    def forward2(self, x):
        # 与forward方法相同的前向传播过程，直到avgpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # 将x展平为一维向量，得到logits
        features = x.view(x.size(0), -1)
        # 不再经过fc层，直接返回logits
        return features
    
    def getFeaturesSize(self):
        return 64


class PreAct_ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet8(num_classes=10, **kwargs):
    model = ResNet_Cifar(
        BasicBlock, [1, 1, 1], num_classes=num_classes, **kwargs)
    return model


def resnet18(num_classes=10, **kwargs):
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], ksize=3, num_classes= num_classes, **kwargs
    )
    return model

def resnet18_mnist(num_classes=10, **kwargs):
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], ksize=7, num_classes= num_classes, in_channels=1, **kwargs
    )
    return model



def resnet20(num_classes=10, **kwargs):
    model = ResNet_Cifar(
        BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
    # model = ResNet(
    #     BasicBlock, [2, 2, 2, 2], ksize=3, num_classes= num_classes, **kwargs
    # )
    return model


def resnet20_brain(num_classes=3, **kwargs):
    model = ResNet_Cifar(
        BasicBlock, [3, 3, 3], num_classes=num_classes, in_channels=1, **kwargs)
    return model


def resnet20_mnist(num_classes=10, **kwargs):
    model = ResNet_mnist(
        BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)
    # model = ResNet(
    #     BasicBlock, [2, 2, 2, 2], ksize=7, in_channels=1, **kwargs
    # )
    return model

def resnet20_defense_mnist(num_classes=10, discriminator=None, T_ft=None, rank_map=None, defense=False, **kwargs):
    model = ResNet_mnist_defense(
        BasicBlock, [3, 3, 3], num_classes=num_classes, discriminator=discriminator, T_ft=T_ft, rank_map=rank_map, defense=defense)





def resnet32(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == "__main__":
    net = resnet20()
    y = net(torch.randn(1, 3, 64, 64))
    print(net)
    print(y.size())
