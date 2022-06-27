# Neural network relief (NNrelief)

## REQUIREMENTS

- Install PyTorch and all requirements using:

      pip install -r requirements.txt



## TRAINING

All other training details that are necessary to obtain pretrained models from experiments can be found in Table 1 of the Appendix.
If you use default settings, you obtain LeNet-5 pretrained model.

- To train the model in the paper, run this command:

```
python train.py --dataset_name <dataset>                           # default=mnist, options = {mnist, cifar10, cifar100}
                --path_data <path to save/load dataset>            # default='./'
                --download_data  <download dataset>                # default=True, options = {True, False}
                --model_name <network architecture to use>         # default=lenet5, options = {lenet5, lenet300-100, 
                                                                                                vgg-like, vgg11, vgg13, vgg16,vgg19, 
                                                                                                resnet20, resnet32, resnet44, resnet56'}
                --path_pretrained_model <path to save model>                     # default=pretrained_model.pth
                --path_init_params <path to initialization parameters>           # default=init_params.pth
                --batch_size <number of examples per batch>                      # default=120
                --train_epochs <number of training epochs>                       # default=60
                --optimizer <optimizer>                                          # default=adam, options={adam, sgd}
                --lr_decay_type <learning rate decay type>                       # default=multistep, options={multistep}
                --lr <initial learning rate>                                     # default=1e-3
                --decay_epochs_train <epochs for multistep decay>                # default=30, options = epoch1 epoch2 ...
                --gamma <multiplicative factor of learning rate decay>           # default=0.1
                --wd <weight decay>                                              # default=5e-4
                --seed <seed>                                                    # default=0
```

### Examples
1) if you want to train LeNet-300-100 model on MNIST with Adam optimizer from the paper, run the following command:

`python train.py --model_name lenet300-100`


2) if you want to train VGG-like model on CIFAR-10 with Adam optimizer from the paper, run the following command:

`python train.py --dataset_name cifar10  --model_name vgg-like --train_epochs 150 --optimizer adam --lr 0.001 --decay_epochs_train 80 120`


3) if you want to train ResNet-20 model on CIFAR-10 with SGD optimizer from the paper, run the following command:

`python train.py --dataset_name cifar10  --model_name resnet20  --train_epochs 150 --optimizer sgd --lr 0.1 --decay_epochs_train 80 120`


4) if you want to train ResNet-56 model on CIFAR-100 with Adam optimizer from the paper, run the following command:

`python train.py --dataset_name cifar100  --model_name resnet56  --train_epochs 200 --optimizer adam --lr 0.001 --decay_epochs_train 120 160`




## EVALUATION

- To evaluate the model, run this command:

```
python eval.py  --dataset_name <dataset>                           # default=mnist, options = {mnist, cifar10, cifar100}
                --path_data <path to save/load dataset>            # default='./'
                --download_data  <download dataset>                # default=True, options = {True, False}
                --model_name <network architecture to use>         # default=lenet5, options = {lenet5, lenet300-100, 
                                                                                                vgg-like, vgg11, vgg13, vgg16, vgg19, 
                                                                                                resnet20, resnet32, resnet44, resnet56}
                --path_pretrained_model <path to a saved model>                  # default=pretrained_model.pth
                --path_init_params <path to initialization parameters>           # default=init_model.pth  # needed only for LeNets
                --alpha_conv <fraction of importance to keep in conv layers>     # default=0.9
                --alpha_fc <fraction of importance to keep in fc layers>         # default=0.95
                --num_iters <number of pruning iterations>                       # default=20
                --prune_batch_size  <number of examples for pruning>             # default=1000
                --train_batch_size  <number of examples per training batch>      # default=120
                --retrain_epochs <number of retraining epochs after pruning>     # default=60
                --optimizer <optimizer>                                          # default=adam, options={adam, sgd}
                --lr_decay_type <learning rate decay type>                       # default=multistep, options={multistep}
                --lr <initial learning rate>                                     # default=1e-3
                --decay_epochs_retrain <epochs for multistep decay>              # default=30, options = epoch1 epoch2 ...
                --gamma <multiplicative factor of learning rate decay>           # default=0.1
                --wd <weight decay>                                              # default=5e-4
                --seed <seed>        
```

### Examples 

1) if you want to prune LeNet-300-100 model (from the folder pretrained_models) on MNIST with hyperparameters from the paper, 
run the following command:

`python eval.py --model_name lenet300-100 --path_pretrained_model pretrained_models/lenet-300-100_pretrained_mnist_seed0.pth 
--path_init_params pretrained_models/lenet-300-100_init_seed0.pth` 


2) if you want to prune VGG-lie model (from the folder pretrained_models) on CIFAR-10 with Adam optimizer and hyperparameters from the paper, 
run the following command:

`python eval.py --model_name vgg-like --path_pretrained_model pretrained_models/VGG-like_pretrained_cifar10_seed0.pth 
--alpha_conv 0.95 --alpha_fc 0.95 --num_iters 6 --retrain_epochs 60 --optimizer adam --lr 0.001 --decay_epochs_train 20 40`


3) if you want to prune ResNet-20 model (from the folder pretrained_models) on CIFAR-10 with SGD optimizer and hyperparameters from the paper, 
run the following command:

`python eval.py --dataset_name cifar10  --model_name resnet20 --path_pretrained_model pretrained_models/resnet20_pretrained_sgd_wd5e-4_cifar10_seed0.pth 
               --alpha_conv 0.95 --alpha_fc 0.99 --num_iters 10 --retrain_epochs 60 --optimizer sgd --lr 0.1 --decay_epochs_train 20 40`

4) if you want to prune ResNet-56 model (from the folder pretrained_models) on CIFAR-100 with Adam optimizer and hyperparameters from the paper, 
run the following command:

`python eval.py --dataset_name cifar100  --model_name resnet56 --path_pretrained_model pretrained_models/resnet56_pretrained_adam_wd5e-4_cifar100_seed0.pth 
               --alpha_conv 0.95 --alpha_fc 0.99 --num_iters 10 --prune_batch_size 2000  --retrain_epochs 80 --optimizer adam --lr 0.001 
               --decay_epochs_train 30 60`
