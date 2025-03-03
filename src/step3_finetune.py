import torch
from utils.config import parser
from datasets.datasets import get_dataset
from models.models import get_model
from step1_defender_EDL import getCsv2, drawFeatures, getFeaturesDataLoader, getFeaturesDataLoaderExtend
from utils.simutils.timer import timer
from utils.helpers import train_epoch_FT, test, gradient_penalty
import torch.optim as optim
import torch
import sys
from drawCross import drawCross

args = parser.parse_args()
if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'

torch.cuda.set_device(0)

def finetune():

    if args.model_tgt == "res20e":
        model_tgt = "res20e"
        # args.model_tgt = "res20e_mnist"
    if  args.model_tgt == "res20e_mnist":
        model_tgt = "res20e_mnist"

    savedir_D = '{}/{}/{}/'.format(args.logdir, args.dataset, model_tgt)
    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, model_tgt)

    train_loader, test_loader = get_dataset(
        args.dataset, args.batch_size, augment=True)

    model = get_model(args.model_tgt, args.dataset, args.pretrained)
    model.id = "server"
    model = model.to(args.device)
    

    D_savepath_wosig = savedir_D + 'D_wosig.pt'
    G_savepath_wosig = savedir + 'G_wosig.pt'
    model_savepath = savedir + 'T.pt'

    features_dim = model.getFeaturesSize()
    
    G = get_model("wgen", features_dim=features_dim, dataset=args.dataset)
    D = get_model("wgen_dis", dataset=args.dataset, features_dim=features_dim)
    G = G.to(args.device)
    D = D.to(args.device)
    

    D.load_state_dict(torch.load(D_savepath_wosig))
    G.load_state_dict(torch.load(G_savepath_wosig))
    model.load_state_dict(torch.load(model_savepath))


    getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
            model=model, fakeDataSet=args.OoDdataset, step='step2', option=2)
    drawFeatures(savedir, args.dataset, features_dim=features_dim, step="step2", option=2)

    getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
            model=model, fakeDataSet=args.OoDdataset, step='step2', option=1)

    drawFeatures(savedir, args.dataset, features_dim=features_dim, step="step2", option=1)

    getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
            model=model, fakeDataSet=args.OoDdataset, step='step2', option=3)

    drawFeatures(savedir, args.dataset, features_dim=features_dim, step="step2", option=3)

    # 4.微调模型

    print("Fine tuning the Model")
    print("The lambda_r is ", args.lambda_r)
    existing_features = []

    # model parameter for finetuning stage
    if args.opt == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr_ft_m,
                        momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, args.epochs_ft_m, last_epoch=-1)
    elif args.opt == 'adam':  # and Adam for the rest
        opt = optim.Adam(model.parameters(), lr=args.lr_ft_m)
    else:
        sys.exit('Invalid optimizer {}'.format(args.opt))

    # WGAN training component

    optimizerG = optim.RMSprop(G.parameters(), lr=args.lr_ft_g)
    optimizerD = optim.RMSprop(D.parameters(), lr=args.lr_ft_d)

    lambda_r = args.lambda_r

    savepath_ft_m = savedir + "T_with_WD_ft.pt"
    savepath_ft_d = savedir + "D_with_WD_ft.pt"
    savepath_ft_g = savedir + "G_with_WD_ft.pt"

    _, original_acc = test(model, args.device, test_loader)
    print("The model before fine tuning", original_acc)

    for epoch in range(args.epochs_ft):

        torch.cuda.empty_cache()
        print('Epoch: {} of fine tuning\n'.format(epoch + 1))

        # 对模型进行训练
        print('===========================\n')
        print('fine tuning the model')

        if (epoch + 1) % 5 == 0: # cifar10 5
            lambda_r = lambda_r * args.lambda_r_decay

        for epoch_M in range(args.epochs_ft_m):

            print("lambda_r = ", lambda_r)

            train_loss, train_acc = train_epoch_FT(model=model, device=args.device, D=D,
                                                   train_loader=train_loader, opt=opt, lambda_r=lambda_r)

            test_loss, test_acc = test(model, args.device, test_loader)
            print('Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f}%\n'.format(
                epoch_M + 1, train_loss, train_acc, test_acc))
            if sch:
                sch.step()

        features_dataloader = getFeaturesDataLoader(
            model=model, train_loader=train_loader, args=args
        )
        
        # 对WGAN进行训练
        print('===========================\n')
        print('fine tuning the WGAN\n')

        for epoch_WGAN in range(args.epochs_ft_wgan):
            for i, data in enumerate(features_dataloader, 0):
                torch.cuda.empty_cache()
                ############################
                # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
                ###########################
                # 使用所有真实样本批次
                # print(i)
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
                noise = torch.randn(
                    b_size, args.latent_dim, device=args.device)

                # 通过生成器生成假样本
                fake = G(noise)

                # 通过判别器分类所有假样本批次
                output = D(fake.detach()).view(-1)

                # 计算损失
                errD_fake = torch.mean(output)

                # 计算梯度惩罚
                gp = gradient_penalty(
                    D, real_cpu.data, fake.data, args=args)

                # 计算判别器的总损失
                errD = errD_real + errD_fake + args.lambda_gp * gp

                # 反向传播
                errD.backward()

                # 更新 D
                optimizerD.step()

                # 对判别器的权重参数进行裁剪，满足WGAN的梯度惩罚
                # for p in D.parameters():
                #     p.data.clamp_(-args.clip, args.clip)

                ############################
                # (2) 更新 G 网络: 最大化 log(D(G(z)))
                ###########################
                # if i % args.n_critic == 0:
                D.eval()
                G.train()
                for j in range(4):

                    noise = torch.randn(
                        b_size, args.latent_dim, device=args.device)

                    # 通过生成器生成假样本
                    fake = G(noise)
                    # print(i, "G")

                    G.zero_grad()
                    # 由于我们刚刚更新了D，需要再次通过D进行前向传播
                    output = D(fake).view(-1)

                    # 计算生成器的损失
                    errG = -torch.mean(output)

                    # 反向传播
                    errG.backward()

                    # 更新生成器
                    optimizerG.step()

        getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                model=model, fakeDataSet=args.OoDdataset, step='step3_' + str(epoch), option=1)

        # 关注 检测指标，看下 FashionMNIST 的 logits 和 Fakelogits 之间拉开的距离
        cr = getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                     model=model, fakeDataSet=args.OoDdataset, step='step3_' + str(epoch), option=2)
        
        getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
            model=model, fakeDataSet=args.OoDdataset, step='step3_' + str(epoch), option=3)

        drawFeatures(savedir, args.dataset, features_dim= features_dim,
                     step="step3_" + str(epoch), option=1)
        drawFeatures(savedir, args.dataset, features_dim= features_dim,
                     step="step3_" + str(epoch), option=2)
        
        drawFeatures(savedir, args.dataset, features_dim= features_dim,
                     step="step3_" + str(epoch), option=3)

    torch.save(model.state_dict(), savepath_ft_m)
    torch.save(D.state_dict(), savepath_ft_d)
    torch.save(G.state_dict(), savepath_ft_g)

    model.load_state_dict(torch.load(savepath_ft_m))
    D.load_state_dict(torch.load(savepath_ft_d))
    G.load_state_dict(torch.load(savepath_ft_g))

    _, test_acc = test(model, args.device, test_loader)

    # 关注模型的 性能维持指标
    print("The model's original accuracy is {:.2f}%\n".format(
        original_acc))
    print(
        "The model's accuracy after fine tuning is {:.2f}%\n".format(test_acc))

    # 关注 WGAN 微调效果
    getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
            model=model, fakeDataSet=args.OoDdataset, step='step3_with_WD', option=1)

    drawFeatures(savedir, args.dataset, features_dim= features_dim, step="step3_with_WD", option=1)


    if args.dataset == "cifar10":
    
        for fakeDataSet in ["svhn", "gtsrb"]:
            print('The fakeDataSet is ' + fakeDataSet)
            # 关注 检测指标，看下FashionMNIST 的 logits 和 Fakelogits 之间拉开的距离
            getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                    model=model, fakeDataSet=fakeDataSet, step='step3_with_WD_' + fakeDataSet, option=2)
            drawFeatures(savedir, args.dataset, features_dim= features_dim,
                         step="step3_with_WD_" + fakeDataSet, option=2)

    if args.dataset == "fashionmnist":
        for fakeDataSet in ["mnist", "kmnist", "mnist_32", "cifar10_gray"]:
            print('The fakeDataSet is ' + fakeDataSet)
            # 关注 检测指标，看下FashionMNIST 的 logits 和 Fakelogits 之间拉开的距离
            getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                    model=model, fakeDataSet=fakeDataSet, step='step3_with_WD_' + fakeDataSet, option=2)
            drawFeatures(savedir, args.dataset, features_dim= features_dim,
                        step="step3_with_WD_" + fakeDataSet, option=2)

    print("==================================")
    drawCross()


def main():
    timer(finetune)
    exit(0)
if __name__ == '__main__':
    main()
