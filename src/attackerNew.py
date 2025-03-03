from attacks import (
    knockoff,
    knockoff_new,
    noise,
    jbda,
    maze,
    maze_new,
)
from utils.helpers import test, test_new
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
from sendMessage import sendMessage

seed = 2021
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy("file_system")

args = parser.parse_args()
torch.cuda.set_device(args.device_id)

# wandb.init(project=args.wandb_project)
run_name = "{}_{}".format(args.dataset, args.attack)
if args.attack == "maze":
    if args.alpha_gan > 0:
        run_name = "{}_{}".format(args.dataset, "pdmaze")
    budget_M = args.budget / 1e6

    if args.white_box:
        grad_est = "wb"
    else:
        grad_est = "nd{}".format(args.ndirs)

    if args.iter_exp > 0:
        run_name += "_{:.2f}M_{}".format(budget_M, grad_est)
    else:
        run_name += "_{:.2f}M_{}_noexp".format(budget_M, grad_est)

# wandb.run.name = run_name
# wandb.run.save()

# Select hardware

if args.device == "gpu":
    import torch.backends.cudnn as cudnn

    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = "cuda"
else:
    args.device = "cpu"


def attack():
    savedir = "{}/{}/{}/".format(args.logdir, args.dataset, args.model_tgt)
    if args.attack == "knockoff":
        savedir_clone = savedir + "clone/"
    elif args.attack == "maze":
        savedir_clone = savedir + "clone1/"
    if not os.path.exists(savedir_clone):
        os.makedirs(savedir_clone)

    train_loader, test_loader = get_dataset(args.dataset, args.batch_size)


    # model after finetuning
    T_ft = get_model(args.model_ft, args.dataset)
    T_ft = T_ft.to(args.device)
    savepathT_ft = savedir + "T_ft.pt"
    T_ft.load_state_dict(torch.load(savepathT_ft))
    features_dim = T_ft.getFeaturesSize()
    T_ft.eval()
    
    # discriminator
    D = get_model("wgen_dis", dataset=args.dataset, features_dim=features_dim)
    D = D.to(args.device)
    savepathD = savedir + "D.pt"
    D.load_state_dict(torch.load(savepathD))
    D.eval()

    rank_map = []
    
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(args.device)
        with torch.no_grad():
            output = T_ft.forward2(inputs)
            prediction = D(output)
        for pred in prediction:
            rank_map.append(pred.item())   
    rank_map.sort()
    T = get_model(args.model_tgt, args.dataset)  # Target (Teacher)

    S = get_model(args.model_clone, args.dataset)  # Clone (Student)
    S = S.to(args.device)

    savepathT = savedir + "T.pt"
    T.load_state_dict(torch.load(savepathT))
    # print(args.device)
    T = T.to(args.device)

    # _, tar_acc = test_new(T, D, T_ft, rank_map, args.device, test_loader)
    _, tar_acc = test(T, args.device, test_loader)
    '''    
    T.Q.to(args.device)
    T.T.to(args.device)
    '''

    print("* Loaded Target Model *")
    print("Target Accuracy: {:.2f}\n".format(tar_acc))

    if args.attack == "noise":
        noise(args, T, S, test_loader, tar_acc)
    elif args.attack == "knockoff":
        # knockoff_new(args, T, T_ft, D, rank_map, S, test_loader, tar_acc)
        knockoff(args, T, S, test_loader, tar_acc)
    elif args.attack == "jbda":
        jbda(args, T, S, train_loader, test_loader, tar_acc)
    elif args.attack == "maze":
        if args.defense == "False":
            maze(args, T, S, train_loader, test_loader, tar_acc)
        else:
            maze_new(args, T, T_ft, D, rank_map, S, train_loader, test_loader, tar_acc)

    else:
        sys.exit("Unknown Attack {}".format(args.attack))

    
    # torch.save(S.state_dict(), savedir_clone + args.model_clone + "/{}.pt".format(args.attack))
    # print("* Saved Sur model * ")



def main():
    pid = os.getpid()
    print("pid: {}".format(pid))
    # try:
    timer(attack)
    # except Exception as e:
    #     sendMessage("实验出错了！")
    #     print(e)
    #     exit(0)
    # sendMessage("实验已经完成！")
    exit(0)


if __name__ == "__main__":
    main()
