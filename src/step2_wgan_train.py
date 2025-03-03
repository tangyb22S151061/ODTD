from step1_defender_EDL import getCsv2, drawFeatures, getFeaturesDataLoader
from models.models import get_model
from datasets.datasets import get_dataset
import torch.optim as optim
import torch
from utils.helpers import gradient_penalty, test
from utils.simutils.timer import timer
from utils.config import parser

args = parser.parse_args()
if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'

torch.cuda.set_device(0)


# 2. logits dataset construct
def train_WGAN(): 

    if args.dataset == "fashionmnist":
        args.OoDdataset = "mnist"
    elif args.dataset == "cifar10":
        args.OoDdataset = "svhn"
    elif args.dataset == "gtsrb":
        args.OoDdataset = "cifar10"
    elif args.dataset == "svhn":
        args.OoDdataset = "cifar10"

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)
    
    model = get_model(args.model_tgt, args.dataset, args.pretrained)
    model.id = "server"
    model = model.to(args.device)

    savepath = savedir + 'T.pt'
    
    model.load_state_dict(torch.load(savepath))
    train_loader, test_loader = get_dataset(
        args.dataset, args.batch_size, augment=True)

    _, original_acc = test(model, args.device, test_loader)

    print("Constructing the features dataset")

    # logits_dataloader = getLogitsDataLoader(
    #     model=model, train_loader=train_loader, args=args)

    features_dataloader = getFeaturesDataLoader(
        model=model, train_loader=train_loader, args=args
    )
    print("==================================")
    
    print("Training the WGAN GP")
    features_dim = model.getFeaturesSize()

    G = get_model("wgen", features_dim=features_dim, dataset=args.dataset)
    D = get_model("wgen_dis", dataset=args.dataset, features_dim=features_dim)
    G = G.to(args.device)
    D = D.to(args.device)

    print("Constructing the features dataset")   
    D.train(), G.train()

    optimizerG = optim.RMSprop(G.parameters(), lr=args.lr_gen_w)
    optimizerD = optim.RMSprop(D.parameters(), lr=args.lr_dis_w)

    for epoch in range(args.epochs_wgan):
        for i, data in enumerate(features_dataloader, 0):
            torch.cuda.empty_cache()
            ############################
            # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
            ###########################
            # 使用所有真实样本批次
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

            # 计算梯度惩罚
            gp = gradient_penalty(D, real_cpu.data, fake.data, args=args)

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
            if i % args.n_critic == 0:
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
            
            
        if epoch % 8 == 0 and args.needEpochPic == True:
            print("Epoch = {}".format(epoch + 1))
            getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                    model=model, fakeDataSet=args.OoDdataset, step='step2' + '_Epoch' + str(epoch), option=1)
            getCsv2(train_loader=train_loader, D=D, G=G, savedir=savedir,
                    model=model, fakeDataSet=args.OoDdataset, step='step2' + '_Epoch' + str(epoch), option=2)
            drawFeatures(savedir, args.dataset, features_dim=features_dim, step="step2" +
                        "_Epoch" + str(epoch), option=1)
            drawFeatures(savedir, args.dataset, features_dim=features_dim, step="step2" +
                        "_Epoch" + str(epoch), option=2)
            

    D_savepath_wosig = savedir + 'D.pt'
    G_savepath_wosig = savedir + 'G.pt'
    torch.save(D.state_dict(), D_savepath_wosig)
    torch.save(G.state_dict(), G_savepath_wosig)




def main():
    timer(train_WGAN)
    exit(0)


if __name__ == '__main__':
    main()


