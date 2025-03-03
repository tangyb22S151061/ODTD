import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import get_nclasses

from scipy.stats import wasserstein_distance


def drawLogits(savedir, dataset, step, option=1):

    # 读取.csv文件
    real_logits = pd.read_csv(savedir + 'real_logits_all_' + step + '.csv')
    # 获取生成logits的图像
    if option == 1:
        fake_logits = pd.read_csv(
            savedir + 'generated_logits_all_' + step + '.csv')

    # 获取虚假数据集的logits的图像
    elif option == 2:
        fake_logits = pd.read_csv(savedir + 'fake_logits_all_' + step + '.csv')

    nclasses = get_nclasses(dataset)
    # 提取第11列数据
    real_data = real_logits.iloc[:, nclasses]
    fake_data = fake_logits.iloc[:, nclasses]

    w_distance = wasserstein_distance(real_data, fake_data)
    print(f"Wasserstein distance between real and fake data: {w_distance}")

    # 四舍五入最小值和最大值
    min_val = np.floor(min(real_data.min(), fake_data.min()))
    max_val = np.ceil(max(real_data.max(), fake_data.max()))

    # 检查最大值和最小值是否相等
    if max_val == min_val:
        print("最大值和最小值相等，无法计算间隔。")
        return

    # 计算间隔
    interval = (max_val - min_val) / 50

    # 定义bins
    bins = np.arange(min_val, max_val, interval)

    # 计算直方图
    real_hist, _ = np.histogram(real_data, bins=bins)
    fake_hist, _ = np.histogram(fake_data, bins=bins)

    # 计算交叉部分的数据量
    cross_count = np.sum(np.minimum(real_hist, fake_hist))

    # 创建直方图
    plt.hist(real_data, bins=bins, alpha=0.5, label='Real')
    if option == 1:
        plt.hist(fake_data, bins=bins, alpha=0.5, label='Generated')
    elif option == 2:
        plt.hist(fake_data, bins=bins, alpha=0.5, label='Fake')

    # 每隔0.1添加一条垂直线
    for x in np.arange(min_val, max_val, interval):
        plt.axvline(x, color='k', linestyle='--', linewidth=0.5)

    # 设置图表标题和标签
    plt.title('Histogram of Logits')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 添加图例
    plt.legend(loc='upper right')

    # 保存图表
    if option == 1:
        plt.savefig(savedir + 'logits_generated&real_dis_' + step + '.png')
    elif option == 2:
        plt.savefig(savedir + 'logits_fake&real_dis_' + step + '.png')
        print("ID,OOD交叉部分的数据量为：", cross_count)
    print("done")
    plt.clf()


def drawFeatures(savedir, dataset, features_dim, step, option=1):

    # 读取.csv文件
    real_features = pd.read_csv(savedir + 'real_features_all_' + step + '.csv')
    # 获取生成features的图像
    if option == 1:
        fake_features = pd.read_csv(
            savedir + 'generated_features_all_' + step + '.csv')

    # 获取虚假数据集的features的图像
    elif option == 2:
        fake_features = pd.read_csv(
            savedir + 'fake_features_all_' + step + '.csv')
    
    elif option == 3:
        fake_features = pd.read_csv(savedir + 'noise_' + step + '.csv')

    nclasses = features_dim
    # 提取第65列数据
    real_data = real_features.iloc[:, nclasses]
    fake_data = fake_features.iloc[:, nclasses]

    if option == 2:
        w_distance = wasserstein_distance(real_data, fake_data)
        print(f"Wasserstein distance between real and fake data: {w_distance}")
    elif option == 3:
        w_distance = wasserstein_distance(real_data, fake_data)

    # 四舍五入最小值和最大值
    min_val = np.floor(min(real_data.min(), fake_data.min()))
    max_val = np.ceil(max(real_data.max(), fake_data.max()))

    # 检查最大值和最小值是否相等
    if max_val == min_val:
        print("最大值和最小值相等，无法计算间隔。")
        return

    # 计算间隔
    interval = (max_val - min_val) / 100

    # 定义bins
    bins = np.arange(min_val, max_val, interval)
    # bins = int(np.sqrt(len(real_data)))

    # 计算直方图
    real_hist, _ = np.histogram(real_data, bins=bins)
    fake_hist, _ = np.histogram(fake_data, bins=bins)

    # 计算交叉部分的数据量
    # cross_count = np.sum(np.minimum(real_hist, fake_hist))
    real_data_min = real_data.min()
    real_data_max = real_data.max()
    cross_count = np.sum((fake_data >= real_data_min) & (fake_data <= real_data_max))

    # 创建直方图
    plt.hist(real_data, bins=bins, alpha=0.5, label='Real')
    if option == 1:
        plt.hist(fake_data, bins=bins, alpha=0.5, label='Generated')
    elif option == 2:
        plt.hist(fake_data, bins=bins, alpha=0.5, label='Fake')
    elif option == 3:
        plt.hist(fake_data, bins=bins, alpha=0.5, label='Noise')

    # 每隔0.1添加一条垂直线
    for x in np.arange(min_val, max_val, interval):
        plt.axvline(x, color='k', linestyle='--', linewidth=0.5)

    # 设置图表标题和标签
    plt.title('Histogram of Features')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 添加图例
    plt.legend(loc='upper right')

    # 保存图表
    if option == 1:
        plt.savefig(savedir + 'features_generated&real_dis_' + step + '.png')
    elif option == 2:
        plt.savefig(savedir + 'features_fake&real_dis_' + step + '.png')
        print("ID,OOD交叉部分的数据量为：", cross_count)
    elif option == 3:
        plt.savefig(savedir + 'features_noise&real_' + step + '.png')
        print("ID,噪声交叉部分的数据量为：", cross_count)
    # print("done")
    plt.clf()

    if option == 2:
        return cross_count
    elif option == 3:
        return w_distance


if __name__ == '__main__':
    savedir = 'logs/fashionmnist/res20e_mnist/'
    drawLogits(savedir=savedir, dataset="fashionmnist",
               step="step3", option=1)
