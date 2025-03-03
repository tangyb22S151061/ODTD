import re
import matplotlib.pyplot as plt


def drawCross(iteration, dataset, model):
    savedir = "/home/tyb/EDL/v2/logs/{}/{}/Ensemble/{}/".format(dataset, model, iteration)
    # 读取文件内容
    with open( savedir + 'result.out', 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'Epoch: (\d+).*?ID,OOD交叉部分的数据量为： (\d+)'

    # 使用re.findall函数来获取所有匹配的结果
    data = re.findall(pattern, content, re.S)  # 添加 re.S 旗标来匹配换行符
    # print(data)
    if not data:
        print("No data found!")


    # 分离Epoch和交叉数据
    epochs, cross_data = zip(*[(int(epoch), int(cross)) for epoch, cross in data])

    # 使用matplotlib绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cross_data)
    plt.xlabel('Epoch')
    plt.ylabel('Cross Data')
    plt.title('Cross Data vs Epoch')
    plt.grid(True)
    plt.savefig(savedir + "Cross Data.png")


    pattern = r'Epoch: (\d+).*?Wasserstein distance between real and fake data: (\d+)'

    # 使用re.findall函数来获取所有匹配的结果
    data = re.findall(pattern, content, re.S)  # 添加 re.S 旗标来匹配换行符
    # print(data)
    if not data:
        print("No data found!")


    # 分离Epoch和交叉数据
    epochs, cross_data = zip(*[(int(epoch), int(cross)) for epoch, cross in data])

    # 使用matplotlib绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, cross_data)
    plt.xlabel('Epoch')
    plt.ylabel('W_Distance')
    plt.title('Wasserstein distance vs Epoch')
    plt.grid(True)
    plt.savefig(savedir + "Wasserstein distance.png")

if __name__ == "__main__":
    for i in range(8):
        drawCross(i)