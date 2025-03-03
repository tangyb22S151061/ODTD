from scipy.stats import wasserstein_distance
from .multitaskLossWrapper import MultiTaskLossWrapper
# from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent
import pandas as pd
import time
import seaborn as sns
import copy
import random
import torch.optim as optim
import itertools
from torch import autograd
from torch.autograd import Variable
import torch
import math
import sys
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
eps = 1e-7
# from ..utils.simutils import logs
# import kornia
import bisect

tanh = nn.Tanh()

def extract_features(model, data, device):
    """
    提取模型生成的特征。
    
    参数:
    model -- 模型对象
    data -- 输入数据
    device -- 设备类型
    
    返回:
    features -- 从模型中提取的特征
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 将数据发送到指定的设备
    data = data.to(device)
    
    # 不计算梯度，因为我们只是在提取特征
    with torch.no_grad():
        features = model.forward2(data)
    
    return features



def CXE_unif(logits):
    # preds = torch.log(logits) # convert to logits
    cxe = -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()
    return cxe

# 待检测性质的黑盒API


def blackbox(x, model, model_posion):

    output = model(x)
    output_softmax = F.softmax(output, dim=1)
    y_max, _ = torch.max(output_softmax, dim=1)

    # 计算α值
    alpha = 1/(1+torch.exp(10000*(y_max.detach()-0.6)))
    alpha = alpha.unsqueeze(1)

    # 计算毒化模型的输出
    output_poison = model_posion(x)

    # 计算新的输出
    y_prime = (1-alpha) * output + alpha * output_poison

    return y_prime

def generate_noise(x):
    output = x
    # 获取top-1预测
    _, top1_pred = output.topk(1, dim=0)

    noise = None
    while True:
        # 生成一个随机的logits张量
        noise = torch.rand_like(output)
        # 获取噪声的top-1预测
        _, noise_top1_pred = noise.topk(1, dim=0)
        # 如果噪声的top-1预测与模型的top-1预测不同，就跳出循环
        if noise_top1_pred != top1_pred:
            break
    return noise

def generate_noise1(x):
    output = x
    # 获取top-1预测
    _, top1_pred = output.topk(1, dim=0)

    noise = None
    while True:
        # 生成一个随机的logits张量
        noise = torch.rand_like(output)
        # 获取噪声的top-1预测
        _, noise_top1_pred = noise.topk(1, dim=0)
        # 如果噪声的top-1预测与模型的top-1预测不同，就跳出循环
        if noise_top1_pred != top1_pred:
            break

    # 生成一个随机排列
    perm = torch.randperm(noise.numel())
    # 使用这个排列来打乱噪声向量
    noise = noise.view(-1)[perm].view(noise.size())

    return noise

def defense_output(data, T, T_ft, D, rank_map):
    min_rank = min(rank_map)
    max_rank = max(rank_map)
    x = T(data)
    pred = D(T_ft.forward2(data))
    x_final = torch.zeros_like(x)
    for i in range(x.size(0)):
        if not min_rank <= pred[i] <= max_rank:
            x_final[i] = generate_noise1(x[i])
        else:
            # lower_ranks = sum([1 for rank in rank_map if pred[i] < rank])
            lower_ranks = bisect.bisect_left(rank_map, pred[i])
            temperature = 0.9 ** (len(rank_map) - lower_ranks)
            x_final[i] = x[i] / (temperature)
    return x_final

def train_epoch(model, device, train_loader, opt, args, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # if args.adv_train:
        #     niter = 10
        #     data_adv = projected_gradient_descent(
        #         model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
        #     output_adv = model(data)
        #     loss += criterion(output_adv, target)

        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc


def train_epoch_AM(model, device, train_loader, train_oe_loader, opt, args, model_poison=None, optimizer_poison=None, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    # tmp
    oe_lamb = 0.1
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)):
        torch.autograd.set_detect_anomaly(True)  # 启用异常检测
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 处理分布外数据
        data_oe, _ = next(iter(train_oe_loader))
        data_oe = data_oe.to(device)
        output_oe = model(data_oe)
        loss_oe = CXE_unif(output_oe)
        loss += loss_oe * oe_lamb

        # if args.adv_train:
        #     niter = 10
        #     data_adv = projected_gradient_descent(
        #         model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
        #     output_adv = model(data)
        #     loss += criterion(output_adv, target)

        # 训练毒化模型
        if model_poison is not None and optimizer_poison is not None:

            inputs_all = torch.cat([data, data_oe])
            outputs_all = model(inputs_all)
            _, targets_all = torch.max(outputs_all.detach(), dim=1)
            outputs_poison = model_poison(inputs_all)
            outputs_poison_softmax = F.softmax(outputs_poison, dim=1)
            outputs_comp = torch.log(1-outputs_poison_softmax + 1e-7)
            loss_poison = criterion(outputs_comp, targets_all)
            loss_poison.backward()
            optimizer_poison.step()

        loss.backward()
        # train_loss += loss
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    model.train()
    return test_loss, test_acc


def test_new(model, D, T_ft, rank_map, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # output = model(data)
            output = defense_output(data, model, T_ft, D, rank_map)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    model.train()
    return test_loss, test_acc


def test_AM(model, device, test_loader, model_posion=None):
    model.eval()
    if model_posion is not None:
        model_posion.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = blackbox(x=data, model=model, model_posion=model_posion)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)

    model.train()
    if model_posion is not None:
        model_posion.train()

    return test_loss, test_acc


def calculate_wasserstein_distance(model, data, D, device):

    features_dim = model.getFeaturesSize()

    # 计算真实样本特征
    data = data.to(device)
    real_features = model.forward2(data)
    real_features = real_features.detach().cpu().numpy()

    # 将真实特征输入到判别器D中
    real_features_tensor = torch.from_numpy(real_features).to(device)
    real_predictions = D(real_features_tensor).detach().cpu().numpy()

    # 计算噪声样本
    num_samples = real_features.shape[0]
    noise = torch.randn(num_samples, features_dim, device=device)

    # 将生成的噪声输入到判别器D中
    noise_tensor = noise.to(device)
    noise_predictions = D(noise_tensor).detach().cpu().numpy()

    # 计算Wasserstein距离
    w_distance = wasserstein_distance(
        real_predictions.ravel(), noise_predictions.ravel())

    return w_distance


def train_epoch_FT(model, D, device, train_loader, opt, lambda_r, disable_pbar=False):
    D.eval()
    model.train()
    model.zero_grad()
    correct = 0
    train_loss = 0
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # 创建MultiTaskLossWrapper实例
    loss_fn_uw = MultiTaskLossWrapper(task_num=2)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        features = model.forward2(data)
        output = model.forward(data)
        # 添加，将最大项修改为最小项相同
        # min_val, _ = torch.min(output, dim=1, keepdim=True)
        # max_val_mask = (output == torch.max(output, dim=1, keepdim=True))
        # max_val_mask = torch.tensor(max_val_mask)
        # max_val_mask = max_val_mask.to(device)
        # output_D = torch.where(max_val_mask, min_val, output)
        # D_l = D(output_D).mean()

        D_l = D(features).mean()
        CE_loss = criterion(output, target)
        # w_distance = calculate_wasserstein_distance(
        #     model, data, D, device)

        # 多任务自适应调整参数
        # loss, _ = loss_fn_uw(CE_loss, D_l)

        # loss, alpha = getMultiObjectLoss(CE_loss, D_l, opt, model)
        # print('alpha is ', alpha)

        loss = CE_loss + lambda_r * D_l

        # print("CE Loss = ", criterion(output, target))
        # print("Regularisation term = ", D_l)
        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc


def train_epoch_FT1(model, D, device, train_loader, opt, lambda_r, disable_pbar=False):
    D.eval()
    model.train()
    model.zero_grad()
    correct = 0
    train_loss = 0
    max_diff = 0  # 初始化最大差距
    max_diff_params = None  # 初始化具有最大差距的参数
    ret_idx = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_fn_uw = MultiTaskLossWrapper(task_num=2)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        features = model.forward2(data)
        output = model.forward(data)

        D_l = D(features).mean()
        CE_loss = criterion(output, target)

        # 多任务自适应调整参数
        loss, _ = loss_fn_uw(CE_loss, D_l)

        # loss = CE_loss + lambda_r * D_l
        loss.backward()
        train_loss += loss.item()

        # 生成与features形状相同的随机噪声
        random_noise = torch.randn_like(features)
        # 计算随机噪声的D(f)
        D_noise = D(random_noise).mean()
        # 计算差距
        diff = torch.abs(D_l - D_noise)

        # 如果这个batch的差距是最大的，更新max_diff_params
        if diff > max_diff:
            max_diff = diff
            max_diff_params = {k: v.clone() for k, v in model.state_dict().items()}
            ret_idx = batch_idx

        opt.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)

    # 如果找到了具有最大差距的参数，更新模型参数
    if max_diff_params:
        model.load_state_dict(max_diff_params)

    return train_loss, train_acc, ret_idx


def train_assosiate(model, D, G, device, train_loader, opt, sch, optD, optG, args, lambda_r, disable_pbar=False):
    # 损失进行平均而非求和
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    # 创建MultiTaskLossWrapper实例
    loss_fn_uw = MultiTaskLossWrapper(task_num=2)


    # 训练模型    
    correct = 0
    train_loss = 0

    epoch_model = args.epochs_ft_m
    epoch_wgan = args.epochs_ft_wgan

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)):
        
        # 1. 开始微调模型
        # 模型微调，让判别器处于预测模式
        D.eval()
        model.train()
        model.zero_grad()

        
        print('Fine tuning the model')
        for epoch in range(epoch_model):

            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            features = model.forward2(data)
            output = model.forward(data)
            D_f = D(features).mean()
            CE_loss = criterion(output, target)

            # 多任务自适应调整参数
            loss, _ = loss_fn_uw(CE_loss, D_f)

            # loss = CE_loss + lambda_r * D_f
            loss.backward()
            train_loss += loss
            opt.step()
            if sch is not None:
                sch.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        correct /= epoch_model
        train_loss /= epoch_model

        # 2. 开始微调WGAN
        print('===========================\n')
        print('Fine tuning the WGAN')
        features = extract_features(model, data, device)
        
        for epoch_gan in range(epoch_wgan):

            ############################
            # 2.1 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            torch.cuda.empty_cache()
            D.train()
            D.zero_grad()
            G.eval() 
            real_cpu = features.to(device)
            # real_cpu = extract_features.to(device)
            b_size = real_cpu.size(0)

            output = D(real_cpu).view(-1)
            errD_real = -torch.mean(output)

            noise = torch.randn(b_size, args.latent_dim, device=device)
            fake = G(noise)
            output = D(fake.detach()).view(-1)
            errD_fake = torch.mean(output)

            gp = gradient_penalty(D, real_cpu.data, fake.data, args=args)
            errD = errD_real + errD_fake + args.lambda_gp * gp
            errD.backward()
            optD.step()

            ############################
            # 2.2 更新 G 网络: 最大化 log(D(G(z)))
            ###########################
            D.eval()
            G.train()
            for j in range(3):
                noise = torch.randn(b_size, args.latent_dim, device = device)
                fake = G(noise)
                G.zero_grad()
                output = D(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                optG.step()

    train_acc = correct * 100. / len(train_loader.dataset)
    train_loss /= len(train_loader)

    return train_loss, train_acc          
            
            
            
def train_wgen_d(D, data, args, G, optimizerD):

    D.train()
    D.zero_grad()
    G.eval()

    # 格式化批次
    real_cpu = data[0].to(args.device)

    b_size = real_cpu.size(0)

    # 向前传播
    output = D(real_cpu).view(-1)

    # 计算损失
    errD_real = -torch.mean(output)

    # 使用所有假样本批次
    # 生成批量的潜在向量
    noise = torch.randn(b_size, args.latent_dim, device=args.device)

    # 通过生成器生成假样本
    fake = G(noise)

    # 通过判别器分类所有假样本批次
    output = D(fake.detach()).view(-1)

    # 计算损失
    errD_fake = torch.mean(output)

    # 计算判别器的总损失
    errD = errD_real + errD_fake

    # 反向传播
    errD.backward()

    # 更新 D
    optimizerD.step()

    # 对判别器的权重参数进行裁剪，满足WGAN的梯度惩罚
    for p in D.parameters():
        p.data.clamp_(-args.clip, args.clip)

    return fake, D


def train_wgen_g(G, D, fake, optimizerG):

    D.eval()
    G.train()
    G.zero_grad()
    # 由于我们刚刚更新了D，需要再次通过D进行前向传播
    output = D(fake).view(-1)

    # 计算生成器的损失
    errG = -torch.mean(output)

    # 反向传播
    errG.backward()

    # 更新生成器
    optimizerG.step()

    return G


def train_WGAN(args, logits_dataloader, D, G, optimizerD, optimizerG):
    for epoch in range(args.epochs_wgan):
        for i, data in enumerate(logits_dataloader, 0):
            ############################
            # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 使用所有真实样本批次

            fake, D = train_wgen_d(D=D, data=data, args=args,
                                   G=G, optimizerD=optimizerD)

            ############################
            # (2) 更新 G 网络: 最大化 log(D(G(z)))
            ###########################
            if i % args.n_critic == 0:
                G = train_wgen_g(G=G, D=D, fake=fake,
                                 optimizerG=optimizerG)

    return D, G


def train_WGAN_FT(args, logits_dataloader, D, G, optimizerD, optimizerG):
    for epoch in range(args.epochs_wgan):
        for i, data in enumerate(logits_dataloader, 0):

            ############################
            # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 使用所有真实样本批次

            fake, D = train_wgen_d(D=D, data=data, args=args,
                                   G=G, optimizerD=optimizerD)

            ############################
            # (2) 更新 G 网络: 最大化 log(D(G(z)))
            ###########################
            if i % (args.epochs_ft_d // args.epochs_ft_g) == 0 or \
                    i % (args.epochs_ft_g // args.epochs_ft_d) == 0:
                G = train_wgen_g(G=G, D=D, fake=fake,
                                 optimizerG=optimizerG)

    return D, G


# 定义梯度惩罚
def gradient_penalty(D, real_data, fake_data, args):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(args.device)
    interpolates = (alpha * real_data + ((1 - alpha)
                    * fake_data)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_data.shape[0], 1).to(args.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def getMultiObjectLoss(loss_1, loss_2, opt, model):
    opt.zero_grad()

    loss_1.backward(retain_graph=True)
    gradient1 = [p.grad.clone() for p in model.parameters()]
    opt.zero_grad()

    loss_2.backward(retain_graph=True)
    gradient2 = [p.grad.clone() for p in model.parameters()]

    alpha = sum((g1.flatten() - g2.flatten()).dot(g2.flatten())
                for g1, g2 in zip(gradient1, gradient2))

    # 计算 (gradient1 - gradient2) 的模长的平方
    norm_square = sum((g1.flatten() - g2.flatten()).norm()**2 for g1,
                      g2 in zip(gradient1, gradient2))

    # 将 alpha 除以 (gradient1 - gradient2) 的模长的平方
    alpha = alpha / norm_square

    # 将 alpha 限制在 [0, 1] 的范围内
    alpha = max(min(alpha, 1), 0)

    loss = alpha * loss_1 + (1 - alpha) * loss_2

    opt.zero_grad()
    return loss, alpha
