import torch
import torch.nn as nn
import os
from scipy.stats import shapiro

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg19o', 'vgg19o_bn', "vgg19_mnist", "vgg19_bn_mnist", 'vgg19p'
]

class VGG19(torch.nn.Module):
    def __init__(self, num_classes, header = "cifar10", use_batchnorm=True):
        super(VGG19, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if header == "cifar10":
            self.classifier = nn.Linear(512, num_classes)

        else:
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_batchnorm(self, channels):
        if self.use_batchnorm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # logits = self.classifier(x.view(-1,512*1*1))
        # # probas = F.softmax(logits,dim = 1)
        # x = x.clone()  # 添加 clone 操作
        logits = self.classifier(x.view(x.size(0), -1))
        return logits
    
    def forward2(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # return x.view(-1, 512*1*1)
        return x.view(x.size(0), -1)
    
    def getFeaturesSize(self):
        return 512

class VGG19_MNIST(torch.nn.Module):
    def __init__(self, num_classes=10, header = "FashionMNIST", use_batchnorm=True):
        super(VGG19_MNIST, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),  # 输入通道数改为1
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if header == "FashionMNIST":
            self.classifier = nn.Linear(512, num_classes)  # 输出节点数改为10

        else:
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_batchnorm(self, channels):
        if self.use_batchnorm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x.view(-1,512*1*1))
        return logits
    
    def forward2(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.view(-1, 512*1*1)

    def getFeaturesSize(self):
        return 512


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward2(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x
    
    def getFeaturesSize(self):
        return 4096
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG19P(torch.nn.Module):
    def __init__(self, num_classes, header = "cifar10", use_batchnorm=True):
        super(VGG19P, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            self.get_batchnorm(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if header == "cifar10":
            self.classifier = nn.Linear(512, num_classes)

        else:
            self.classifier = nn.Sequential(
                nn.Linear(512*1*1, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )

        self.id = 'server'
        '''
            功能:定义增长集、最小距离集合、各类别阈值等参数，用以检测        
        '''
        self.D = []
        self.G = [[] for _ in range(num_classes)]
        self.DG = [[] for _ in range(num_classes)]
        self.attack = False
        self.Thre = [[torch.tensor(0)] for _ in range(num_classes)]
        self.delta = 0.9

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_batchnorm(self, channels):
        if self.use_batchnorm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()

    def forward(self,x):
        
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)

        t = t.view(x.size(0), -1)
        t = self.classifier(t)

        if self.id == 'user' and self.training == False:
            if self.attack == True:
                return 'Detected by PRADA'
            # 遍历batch中的每个样本
            batch_size = x.size(0)
            for i in range(batch_size) :
                # c ← F(x)
                sample = x[i]
                _, c = torch.max(t[i], dim=0)
                
                if not self.G[c]:
                    self.G[c].append(x[i])
                    self.DG[c].append(torch.tensor(0,device='cuda'))
                else:
                    d = []
                    for y in self.G[c]:
                        d.append(torch.dist(x[i], y, p=2))
                    
                    dmin = min(d)
                    dmin = torch.tensor(dmin)
                    self.D.append(dmin)
                    
                    if dmin > self.Thre[c][0]:
                        self.G[c].append(x[i])
                        self.DG[c].append(dmin.to('cuda'))
                        DG = torch.stack(self.DG[c])
                        bar = torch.mean(DG)
                        std = torch.std(DG)
                        self.Thre[c][0] = torch.max(self.Thre[c][0], bar - std)
                
                #analyze distribution for D
                if len(self.D) > 100:
                    # D'
                    D1 = []
                    
                    # 获取关于距离列表的均值和标准差
                    D_tensor = torch.stack(self.D)
                    bar = torch.mean(D_tensor)
                    std = torch.std(D_tensor)

                    # 获得D'
                    for z in self.D :
                        if z > bar - 3*std and z < bar + 3*std:
                            D1.append(z)
                    
                    D1 = torch.stack(D1).cpu().numpy()
                    W  = shapiro(D1)[0]
                    if W < self.delta :
                        self.attack = True
                        return 'Detected by PRADA'
                    else:
                        self.attack = False
            if self.attack == True:
                return 'Detected by PRADA'
        return t
    
    def forward2(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # return x.view(-1, 512*1*1)
        return x.view(x.size(0), -1)
    
    def getFeaturesSize(self):
        return 512




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, device, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, device='cpu', **kwargs)


def vgg11_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, device, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, device='cpu', **kwargs)


def vgg13_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, device, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, device='cpu', **kwargs)


def vgg16_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, device, **kwargs)


def vgg19(num_classes, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return VGG19(num_classes=num_classes, use_batchnorm=False)
    # def vgg19(pretrained=False, progress=True, device="cuda:0", **kwargs):  
    # return _vgg('vgg19', 'E', False, pretrained, progress, device=device, **kwargs)

def vgg19p(num_classes, **kwargs):
    return VGG19P(num_classes=num_classes, use_batchnorm=False)

def vgg19_mnist(num_classes, **kwargs):
    return VGG19_MNIST(num_classes=num_classes, use_batchnorm=False)

def vgg19o(pretrained=False, progress=True, device="cuda:0", **kwargs):  
    return _vgg('vgg19', 'E', False, pretrained, progress, device=device, **kwargs)

def vgg19o_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, device, **kwargs)

def vgg19_bn(num_classes, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # def vgg19_bn(pretrained=False, progress=True, device='cpu', **kwargs):
    # return _vgg('vgg19_bn', 'E', True, pretrained, progress, device, **kwargs)
    return VGG19(num_classes=num_classes, use_batchnorm=True)

def vgg19_bn_mnist(num_classes, **kwargs):
    return VGG19_MNIST(num_classes=num_classes, use_batchnorm=True)

