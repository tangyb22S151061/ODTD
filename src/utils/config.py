import argparse

parser = argparse.ArgumentParser(
    description="MAZE: Model Stealing attack using Zeroth order gradient Estimation"
)

# Shared
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="mnist/fashionmnist/cifar10/cifar100/svhn/gtsrb",
)

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
parser.add_argument(
    "--logdir", type=str, default="./logs", help="Path to output directory"
)
parser.add_argument("--device", type=str, default="gpu", help="cpu/gpu/tpu")
parser.add_argument("--ngpu", type=int, default=1, help="Num GPUs to use")
parser.add_argument("--device_id", type=int, default=0)

# Defender
parser.add_argument(
    "--lr_tgt", type=float, default=0.1, help="Learning Rate of Target Model"
)
parser.add_argument(
    "--model_tgt", type=str, default="res20", help="Target Model[res20/wres22/conv3]"
)
parser.add_argument("--pretrained", action="store_true",
                    help="Use Pretrained model")
parser.add_argument("--adv_train", action="store_true",
                    help="Use adversarial training")
parser.add_argument("--eps_adv", type=float, default=0.1,
                    help="epsilon for adv train")

# Attacker
parser.add_argument("--white_box", action="store_true",
                    help="assume white box target")
parser.add_argument(
    "--attack",
    type=str,
    default="maze",
    help="Attack Type [knockoff/abm/zoo/knockoff_zoo/knockoff_augment]",
)
parser.add_argument("--opt", type=str, default="sgd",
                    help="Optimizer [adam/sgd]")
parser.add_argument(
    "--model_clone", type=str, default="wres22", help="Clone Model [res20/wres22/conv3]"
)
parser.add_argument(
    "--model_gen", type=str, default="conv3_gen", help="Generator Model [conv3_gen]"
)
parser.add_argument(
    "--model_dis", type=str, default="conv3_dis", help="Discriminator Model [conv3_dis]"
)
parser.add_argument(
    "--lr_clone", type=float, default=0.1, help="Learning Rate of Clone Model"
)
parser.add_argument(
    "--latent_dim", type=int, default=128, help="dimensionality of latent vector"
)

parser.add_argument("--defense", default=True, 
                    help="the model of fine tuning")

# KnockoffNets
parser.add_argument(
    "--dataset_sur",
    type=str,
    default="cifar100",
    help="mnist/fashionmnist/cifar10/cifar100/svhn/gtsrb",
)

# MAZE
parser.add_argument(
    "--budget", type=float, default=3e7, metavar="N", help="Query Budget for Attack"
)
parser.add_argument(
    "--log_iter", type=float, default=1000, metavar="N", help="log frequency"
)

parser.add_argument(
    "--lr_gen", type=float, default=1e-4, help="Learning Rate of Generator Model"
)
parser.add_argument(
    "--lr_dis", type=float, default=1e-4, help="Learning Rate of Discriminator Model"
)
parser.add_argument(
    "--eps", type=float, default=1e-3, help="Perturbation size for noise"
)
parser.add_argument(
    "--ndirs", type=int, default=10, help="Number of directions for MAZE"
)
parser.add_argument(
    "--mu", type=float, default=0.001, help="Smoothing parameter for MAZE"
)

parser.add_argument(
    "--iter_gen", type=int, default=1, help="Number of iterations of Generator"
)
parser.add_argument(
    "--iter_clone", type=int, default=5, help="Number of iterations of Clone"
)
parser.add_argument(
    "--iter_exp", type=int, default=10, help="Number of Exp Replay iterations of Clone"
)

parser.add_argument(
    "--lambda1", type=float, default=10, help="Gradient penalty multiplier"
)
parser.add_argument("--disable_pbar", action="store_true",
                    help="disable progress bar")
parser.add_argument(
    "--alpha_gan", type=float, default=0.0, help="Weight given to gan term"
)


parser.add_argument(
    "--wandb_project", type=str, default="maze", help="wandb project name"
)


# JBDA
parser.add_argument(
    "--aug_rounds", type=int, default=6, help="Number of augmentation rounds for JBDA"
)
parser.add_argument(
    "--num_seed", type=int, default=100, help="Number of seed examples for JBDA"
)

# noise
parser.add_argument(
    "--noise_type",
    type=str,
    default="ising",
    choices=["ising", "uniform"],
    help="noise type",
)

# Extra
parser.add_argument(
    "--iter_gan", type=float, default=0, help="Number of iterations of Generator"
)
parser.add_argument(
    "--iter_log_gan",
    type=float,
    default=1e4,
    metavar="N",
    help="log frequency for GAN training",
)
parser.add_argument(
    "--iter_dis", type=int, default=5, help="Number of iterations of Discriminator"
)
parser.add_argument("--load_gan", action="store_true", help="load gan")


# 适应于AM的参数

parser.add_argument("--dataset_oe", default="kmnist",
                    help="mnist/fashionmnist/cifar10/cifar100/svhn/gtsrb/kmnist")

# EDL相关超参数
# 第二阶段 性能训练
parser.add_argument("--lr_gen_w", default=4e-4,
                    help="the learning rate of the WGAN's generator G")
parser.add_argument("--lr_dis_w", default=8.8e-5,
                    help="the learning rate of the WGAN's discriminator D")
parser.add_argument("--n_critic", default=4,  # 4
                    help="the number of training times of WGAN's discriminator D before training WGAN's generator G")
parser.add_argument("--epochs_wgan", default=256,
                    help="The training epoch of WGAN")
# parser.add_argument("--clip", default=0.06,
#                     help="the truncation parameter of the WGAN's discriminator D")
parser.add_argument("--lambda_gp", default=10,
                    help="the truncation parameter of the WGAN's discriminator D")
parser.add_argument("--needEpochPic", default=True,
                    help="Draw the logits of generated logits and the real logits in the different Epoch")


# 第三阶段 M、G、D轮流训练微调
parser.add_argument("--OoDdataset", default="mnist")
parser.add_argument("--lambda_r", default=4,
                    help="the regularization coefficient")
parser.add_argument("--decay_epoch", default=5,
                    help="the decay of epoch for decaying")
parser.add_argument("--lambda_r_decay", default=0.5,
                    help="the decay factor of regularization coefficient")

parser.add_argument("--lr_ft_m", default=1e-3, # 6e-3
                    help="the learning rate of the model while fine tuning")
parser.add_argument("--lr_ft_g", default=1e-5,
                    help="the learning rate of the G while fine tuning")
parser.add_argument("--lr_ft_d", default=9e-6,  # 9e-6
                    help="the learning rate of the D while fine tuning")

parser.add_argument("--epochs_ft", default=10,
                    help="the epochs of total fine tuning stage")
parser.add_argument("--epochs_ft_m", default=1,
                    help="the epochs of model training in the fine tuning stage")
# parser.add_argument("--epochs_ft_g", default=10,
#                     help="the epochs of WGAN's generator G in the fine tuning stage")
# parser.add_argument("--epochs_ft_d", default=10,
#                     help="the epochs of WGAN's discriminator D training in the fine tuning stage")

parser.add_argument("--epochs_ft_wgan", default=5,
                    help="the epochs of WGAN's discriminator D training in the fine tuning stage")

parser.add_argument("--model_ft", default=None, 
                    help="the model of fine tuning")



