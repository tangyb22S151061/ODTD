import torch.cuda
import pandas as pd
from drawLogits import drawLogits, drawFeatures
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import csv
from tqdm import tqdm
from utils.helpers import train_epoch, train_epoch_FT, test, gradient_penalty
from datasets import get_dataset
from models.models import get_model
from utils.config import parser
from utils.simutils.timer import timer
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import numpy as np
import scipy
seed = 2020
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy('file_system')

# from utils.helpers import test, train_epoch

args = parser.parse_args()

if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'

torch.cuda.set_device(args.device_id)


'''
    函数功能:
        训练完D后
        (1) 真实logits以及对应D的判别结果存入.csv文件中 
        (2) 生成logits以及对应G的判别结果存入.csv文件中
        (3) 分布外样本logits以及对应G的判别结果存入.csv文件中
'''


def getCsv(train_loader, D, G, savedir, model, fakeDataSet, step, option):

    # 1. 用以向训练好的判别器D输入logits，将对应的logits以及D的判别结果存至.csv文件中
    # 选择全部的真实logits

    D.eval()
    G.eval()
    model.eval()

    logits_list = []
    for data, _ in train_loader:

        data = data.to(args.device)

        with torch.no_grad():
            logits = model(data)
        logits_list.append(logits.cpu().numpy())

    logits_array = np.concatenate(logits_list, axis=0)

    real_logits_selected = logits_array

    # 将真实logits输入到判别器D中
    real_logits_tensor = torch.from_numpy(real_logits_selected).to(args.device)
    with torch.no_grad():
        real_predictions = D(real_logits_tensor).cpu().numpy()

    # 将预测结果并入到logits向量中
    real_logits_selected = np.hstack(
        (real_logits_selected, real_predictions.reshape(-1, 1)))

    # 创建DataFrame
    real_df = pd.DataFrame(real_logits_selected)

    # 将真实logits写入.csv文件
    real_df.to_csv(savedir + 'real_logits_all_' + step + '.csv', index=False)

    # 2. 用训练好的生成器G生成和真实logits数量相同的虚假logits，送入判别器中，将对应logits以及D的判别结果存至.csv文件中
    # 生成GAN logits

    if option == 1:
        num_samples = real_logits_selected.shape[0]
        noise = torch.randn(num_samples, args.latent_dim, device=args.device)
        with torch.no_grad():
            generated_logits = G(noise).cpu().numpy()

        # 将生成的logits输入到判别器D中
        generated_logits_tensor = torch.from_numpy(
            generated_logits).to(args.device)
        with torch.no_grad():
            generated_predictions = D(generated_logits_tensor).cpu().numpy()

        # 将预测结果并入到logits向量中
        generated_logits = np.hstack(
            (generated_logits, generated_predictions.reshape(-1, 1)))

        # 创建DataFrame
        generated_df = pd.DataFrame(generated_logits)

        # 将生成的logits写入.csv文件
        generated_df.to_csv(
            savedir + 'generated_logits_all_' + step + '.csv', index=False)

    # 3.用分布外样本访问model，获得对应的logits
    # 获取分布外样本的dataloader
    elif option == 2:
        fake_loader, test_loader = get_dataset(
            fakeDataSet, args.batch_size, augment=True)
        # 初始化一个空的列表来保存logits和判别器的预测
        data_list = []

        # 遍历MNIST数据集
        for images, _ in fake_loader:
            images = images.to(args.device)

            # 通过原始模型获取logits
            with torch.no_grad():
                logits = model(images)

            # 将logits输入到判别器中
            D_pred = D(logits).detach().cpu().numpy()

            # 将logits和判别器的预测一起保存
            for logit, pred in zip(logits.detach().cpu().numpy(), D_pred):
                pred = pred.squeeze()
                data_list.append(np.concatenate([logit, [pred]]))

        # 将数据保存到CSV文件中
        df = pd.DataFrame(data_list)
        df.to_csv(savedir + 'fake_logits_all_' + step + '.csv', index=False)

    return


def getCsv2(train_loader, D, G, savedir, model, fakeDataSet, step, option, normalization=False):

    # 1. 用以向训练好的判别器D输入features，将对应的features以及D的判别结果存至.csv文件中
    # 选择全部的真实features
    features_dim = model.getFeaturesSize()
    D.eval()
    G.eval()
    model.eval()

    features_list = []
    for data, _ in train_loader:

        data = data.to(args.device)

        with torch.no_grad():
            # 调用model.forward2(data)来获取features
            features = model.forward2(data)
        features_list.append(features.cpu().numpy())

    features_array = np.concatenate(features_list, axis=0)

    real_features_selected = features_array

    # 将真实features输入到判别器D中
    real_features_tensor = torch.from_numpy(
        real_features_selected).to(args.device)
    
    with torch.no_grad():
        real_predictions = D(real_features_tensor).cpu().numpy()
        if normalization:
            real_predictions = 1 / (1 + np.exp(-real_predictions))

    real_median = np.median(real_predictions)
    real_std = np.std(real_predictions)

    # 将预测结果并入到features向量中
    real_features_selected = np.hstack(
        (real_features_selected, real_predictions.reshape(-1, 1)))
    
    # 创建DataFrame
    real_df = pd.DataFrame(real_features_selected)

    # 将真实features写入.csv文件
    real_df.to_csv(savedir + 'real_features_all_' + step + '.csv', index=False)

    # 2. 用训练好的生成器G生成和真实features数量相同的虚假features，送入判别器中，将对应features以及D的判别结果存至.csv文件中
    # 生成GAN features

    if option == 1:
        num_samples = real_features_selected.shape[0]
        noise = torch.randn(num_samples, args.latent_dim, device=args.device)
        with torch.no_grad():
            generated_features = G(noise).cpu().numpy()

        # 将生成的features输入到判别器D中
        generated_features_tensor = torch.from_numpy(
            generated_features).to(args.device)
        with torch.no_grad():
            generated_predictions = D(generated_features_tensor).cpu().numpy()
            if normalization:
                generated_predictions = 1 / (1 + np.exp(-generated_predictions))

        # 将预测结果并入到features向量中
        generated_features = np.hstack(
            (generated_features, generated_predictions.reshape(-1, 1)))

        # 创建DataFrame
        generated_df = pd.DataFrame(generated_features)

        # 将生成的features写入.csv文件
        generated_df.to_csv(
            savedir + 'generated_features_all_' + step + '.csv', index=False)

    # 3.用分布外样本访问model，获得对应的features
    # 获取分布外样本的dataloader
    elif option == 2:
        fake_loader, test_loader = get_dataset(
            fakeDataSet, args.batch_size, augment=True)
        # 初始化一个空的列表来保存features和判别器的预测
        data_list = []

        # 遍历MNIST数据集
        for images, _ in fake_loader:
            images = images.to(args.device)

            # 通过原始模型获取features
            with torch.no_grad():
                features = model.forward2(images)
                # 将features输入到判别器中
                D_pred = D(features).detach().cpu().numpy()
                if normalization:
                    D_pred = 1 / (1 + np.exp(-D_pred))

            # 将features和判别器的预测一起保存
            for feature, pred in zip(features.detach().cpu().numpy(), D_pred):
                pred = pred.squeeze()
                data_list.append(np.concatenate([feature, [pred]]))

        data_array = np.array(data_list)
        fake_median = np.median(data_array[:, -1])
        fake_std = np.std(data_array[:, -1])

        # 将数据保存到CSV文件中
        df = pd.DataFrame(data_list)
        df.to_csv(savedir + 'fake_features_all_' + step + '.csv', index=False)

        return abs(fake_median - real_median) - real_std - fake_std
    

    # 用随机噪声访问
    elif option == 3:
        num_samples = real_features_selected.shape[0]
        noise = torch.randn(num_samples, features_dim, device=args.device).cpu().numpy()

        # 将生成的噪声输入到判别器D中
        noise_tensor = torch.from_numpy(noise).to(args.device)
        with torch.no_grad():
            noise_predictions = D(noise_tensor).cpu().numpy()
            if normalization:
                noise_predictions = 1 / (1 + np.exp(-noise_predictions))

        # 将预测结果并入到噪声向量中
        noise = np.hstack((noise, noise_predictions.reshape(-1, 1)))

        # 创建DataFrame
        noise_df = pd.DataFrame(noise)

        # 将生成的噪声写入.csv文件
        noise_df.to_csv(savedir + 'noise_' + step + '.csv', index=False)

def getLogitsDataLoader(model, train_loader, args):
    model.eval()
    logits_list = []

    for i, (data, _) in enumerate(train_loader):

        # if i > 50:
        #     break

        data = data.to(args.device)

        with torch.no_grad():
            logits = model(data)
        logits_list.append(logits.cpu().numpy())

    logits_array = np.concatenate(logits_list, axis=0)
    # np.save(savedir+ 'logits.npy', logits_array)

    logits_tensor = torch.from_numpy(logits_array)
    logits_dataset = TensorDataset(logits_tensor)
    logits_dataloader = DataLoader(
        dataset=logits_dataset, batch_size=args.batch_size, shuffle=True)

    return logits_dataloader


def getFeaturesDataLoader(model, train_loader, args):
    model.eval()
    features_list = []

    for i, (data, _) in enumerate(train_loader):

        data = data.to(args.device)

        with torch.no_grad():
            # 调用model.forward2(data)来获取features
            features = model.forward2(data)
        features_list.append(features.cpu().numpy())

    features_array = np.concatenate(features_list, axis=0)
    # np.save(savedir+ 'features.npy', features_array)

    features_tensor = torch.from_numpy(features_array)
    features_dataset = TensorDataset(features_tensor)
    features_dataloader = DataLoader(
        dataset=features_dataset, batch_size=args.batch_size, shuffle=True)

    return features_dataloader

def getFeaturesDataLoaderExtend(model, train_loader, args, existing_features=None):
    model.eval()
    features_list = existing_features if existing_features is not None else []

    for i, (data, _) in enumerate(train_loader):
        data = data.to(args.device)

        with torch.no_grad():
            features = model.forward2(data)
        features_list.append(features.cpu().numpy())
    features_array = np.concatenate(features_list, axis=0)
    features_tensor = torch.from_numpy(features_array)
    features_dataset = TensorDataset(features_tensor)
    features_dataloader = DataLoader(
        dataset=features_dataset, batch_size=args.batch_size, shuffle=True)

    return features_dataloader, features_list



def train_defender():

    # print("lambda = ", args.lambda_r)
    print("=======================================")

    # 1. Original model training
    print("Training the original model")
    # if args.model_tgt == "res20e":
    #     model_tgt = "res20e"
    # elif args.model_tgt == "res20e_mnist":
    #     model_tgt = "res20e_mnist"
    # elif args.model_tgt == "vgge":
    #     model_tgt = "vgge"
        
    # args.model_tgt = "res20e_mnist"
        
    model = get_model(args.model_tgt, args.dataset, args.pretrained)
    print(model)
    model.id = "server"
    model = model.to(args.device)
    train_loader, test_loader = get_dataset(
        args.dataset, args.batch_size, augment=True)

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = savedir + 'T.pt'

    sch = None
    if args.opt == 'sgd':  # Paper uses SGD with cosine annealing for CIFAR10
        opt = optim.SGD(model.parameters(), lr=args.lr_tgt,
                        momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, args.epochs, last_epoch=-1)
    elif args.opt == 'adam':  # and Adam for the rest
        opt = optim.Adam(model.parameters(), lr=1e-5)
        
    else:
        sys.exit('Invalid optimizer {}'.format(args.opt))

    # 初始化最高测试准确率为0
    best_acc = 0

    test_acc_history = []
    train_loss_history = []
    
    for epoch in range(args.epochs):

        train_loss, train_acc = train_epoch(
            model, args.device, train_loader, opt, args)
        test_loss, test_acc = test(model, args.device, test_loader)
        print('Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f}%\n'.format(
            epoch+1, train_loss, train_acc, test_acc))
        
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss.cpu().detach().numpy())

        if sch:
            sch.step()
        # 如果当前测试准确率大于最高测试准确率
        if test_acc > best_acc:
            # 更新最高测试准确率
            best_acc = test_acc
            # 保存当前模型参数
            torch.save(model.state_dict(), savepath)
    
    # 绘制折线图
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_acc_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Test Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # 保存图像到指定文件夹中
    plt.savefig(os.path.join(savedir, 'loss_acc.png'))

    model.load_state_dict(torch.load(savepath))
    _, original_acc = test(model, args.device, test_loader)
    print('original acc = ', original_acc)
    print("==================================")


def main():
    timer(train_defender)
    exit(0)


if __name__ == '__main__':
    main()
