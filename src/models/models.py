import torch
import sys
import torch.nn as nn
import os.path as osp
import torchvision.models as models
import torch.nn.functional as F


from . import (
    conv3,
    lenet,
    wresnet,
    resnet,
    resnet_DDT,
    resnet_PRADA,
    resnet_AM,
    resnet_ODTD,
    conv3_gen,
    conv3_cgen,
    conv3_dis,
    conv3_mnist,
    wgen,
    wgen_dis
)
from .cifar10_models import resnet18
from .cifar10_models.vgg import *

import sys
from datasets import get_nclasses


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_dict = {

    "conv3_gen": conv3_gen.conv3_gen,
    "conv3_cgen": conv3_cgen.conv3_cgen,
    "conv3_dis": conv3_dis.conv3_dis,
    "lenet": lenet.lenet,
    "conv3": conv3.conv3,
    "conv3_mnist": conv3_mnist.conv3_mnist,
    "wres22": wresnet.WideResNet,

    # w/o def
    "res20": resnet.resnet20,
    "res20_mnist": resnet.resnet20_mnist,

    # PVMTA
    "res20d": resnet_DDT.resnet20_mnist,
    "res20d_cifar": resnet_DDT.resnet20,

    # prada's def
    "res20p_mnist": resnet_PRADA.resnet20_mnist,
    "res20p": resnet_PRADA.resnet20,
    "res18p": resnet_PRADA.resnet18,
    "res18p_mnist": resnet_PRADA.resnet18_mnist,
    "vgg19p": vgg19p,

    # am's def
    "res20a": resnet_AM.resnet20,
    "res20a_mnist": resnet_AM.resnet20_mnist,
    "res18a": resnet_AM.resnet18,
    "res18a_mnist": resnet_AM.resnet18_mnist,

    "res18_ptm": resnet18,
    "vgg13_bn": vgg13_bn,

    # wgan
    "wgen": wgen.wgen,
    "wgen_dis": wgen_dis.wgen_dis,

    # edl def
    "res20o": resnet_ODTD.resnet20,
    "res20o_mnist": resnet_ODTD.resnet20_mnist,

    "res18o": resnet_ODTD.resnet18,
    "res18o_mnist": resnet_ODTD.resnet18_mnist,  

    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_mnist": vgg19_mnist,
    "vgg19a": vgg19,
    
    "vgg19_bn": vgg19_bn,
    
    "vgg19o": vgg19o,
    "vgg19o_bn": vgg19o_bn,
    
}

gen_channels_dict = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "fashionmnist": 1,
}

gen_dim_dict = {
    "cifar10": 8,
    "cifar100": 8,
    "gtsrb": 8,
    "svhn": 8,
    "mnist": 7,
    "fashionmnist": 7,
}

in_channel_dict = {
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "mnist": 1,
    "fashionmnist": 1,
}


def get_model(modelname, dataset="", pretrained=None, latent_dim=10, features_dim=256, alpha=torch.tensor(0.6), Q_thre=torch.tensor(50), M_thre=torch.tensor(20), **kwargs):
    model_fn = model_dict[modelname]
    num_classes = get_nclasses(dataset)

    if modelname in [
        "conv3",
        "lenet",
        "res20",
        "res20_mnist",
        "conv3_mnist",
        "res20p_mnist",
        "res20p",
        "res18p_mnist",
        "res18p",
        "res20a",
        "res20a_mnist",
        "res18a",
        "res18a_mnist",
        "res20o_mnist",
        "res20o",
        "res18o_mnist",
        "res18o",
        "res18",
        "res18_mnist"
    ]:
        model = model_fn(num_classes)
    # elif modelname == "res20e":
    #     model = model_fn()
    elif modelname in [
        "res20d",
        "res20d_cifar"
    ]:
        model = model_fn(num_classes, alpha=alpha,
                         Q_thre=Q_thre, M_thre=M_thre)

    elif modelname == "wres22":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=22,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=22, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname in ["conv3_gen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
        )
    elif modelname in ["conv3_cgen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
            n_classes=num_classes,
        )
    elif modelname in ["conv3_dis"]:
        model = model_fn(channels=gen_channels_dict[dataset], dataset=dataset)
    elif modelname in ["res18_ptm"]:
        model = model_fn(pretrained=pretrained)
    elif modelname in ["wgen"]:
        model = model_fn(
            output_dim=features_dim,
        )
    elif modelname in ["wgen_dis"]:
        model = model_fn(input_dim=features_dim).to("cuda")

    elif modelname in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19o', 'vgg19o_bn']:
        model = model_fn()
    elif modelname in ['vgg19', 'vgg19_bn', 'vgg19_mnist', 'vgg19_bn_mnist', 'vgg19a', 'vgg19p']:
        model = model_fn(num_classes=num_classes)
    else:
        sys.exit("unknown model")

    return model
