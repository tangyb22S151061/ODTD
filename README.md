# ODTD: Protect Model from Extraction Attacks with Out of Distribution Detection and Temperature Decay

Implementation of paper *"ODTD: Protect Model from Extraction Attacks with Out of Distribution Detection and Temperature Decay"*

# Requirements

* Python 3.8.12
* Pytorch 1.10.0

The environment can be set up as:
```bash
$ pip install -r requirements.txt       # pip
```

# Usage

### Train the defend model

#### Step I. Train the original model

- You can train the original model by running the following prompt:

    >  python src/step1_defender_EDL.py --dataset=fashionmnist --model_tgt=res20o_mnist --epochs=100

#### Step II. Train the WGAN

- You can train the WGAN by running the following prompt:

    > python src/step2_wgan_train.py --dataset=fashionmnist --model_tgt=res20o_mnist --epochs_wgan=128

#### Step III. Assosiate Fine-tune

- You can implement the assoisate finetuning by running the following prompt:

    > python src/step3_finetune.py --dataset=fashionmnist --model_tgt=res20o_mnist --epochs_ft=10

### Launch the attack

- After training a defender model (which will be stored in logs folder), you can lauch the attack between "Knockoff" and "MAZE". You can lauch the attack by running the following prompt:

    > python src/attacker.py --dataset=fashionmnist --model_tgt=res20_mnist --model_clone=wres22 --attack=maze --budget=3e7 --lr_clone=0.1 --lr_gen=1e-3 --iter_clone=5 --iter_exp=10

    > python src/attacker.py --dataset=fashionmnist --model_tgt=res20_mnist --model_clone=wres22 --attack=knockoff --dataset_sur=mnist --epochs=100

- You can change the dataset_sur (which can be selected from the above datasets) to lauch the Knockoff, also you can change the budget of MAZE to control its allowed query counts. To attack different defender models, you can change the model_tgt, which has been decided when you train it. As for the MAZE attack, you can also change the hyper-parameters lr_clone, lr_gen, iter_clone, iter_exp 
to change its setting.

- Especially, for AM defender model, you can lauch the attack by running the following prompt:
    > python src/attacker_AM.py --dataset=fashionmnist --model_tgt=res20a_mnist --model_clone=wres22 --attack=knockoff --dataset_sur=mnist --epochs=100

 # REFERENCES
 
 *[https://github.com/JonasGeiping/breaching](https://github.com/sanjaykariyappa/MAZE)https://github.com/sanjaykariyappa/MAZE*
