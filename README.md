# Neural network relief (NNrelief)

## REQUIREMENTS

- Install PyTorch and all requirements using:

      pip install -r requirements.txt



## TRAINING

All other training details that are necessary to obtain pretrained models from experiments can be found in Table 1 of the Appendix.
If you use default settings, you obtain LeNet-5 pretrained model.

- To train the model in the paper, run this command:


python train.py --dataset_name <dataset> \                         # default=mnist, options = {mnist, cifar10, cifar100}
                --path_data <path to save/load dataset> \          # default='./'
                --download_data  <download dataset>  \             # default=True, options = {True, False}
                --model_name <network architecture to use>         # default=lenet5, options = {lenet5, lenet300-100, vgg-like, vgg11, vgg13, 
                                                                                                  vgg16,vgg19, resnet20, resnet32, resnet44, 
                                                                                                  resnet56'}
                --path_pretrained_model <path to save model> \                # default=pretrained_model.pth
                --path_init_params <path to initialization parameters> \         # default=init_params.pth
                --batch_size <number of examples per batch> \      # default=120
                --train_epochs <number of training epochs> \       # default=60
                --optimizer <optimizer> \                          # default=adam, options={adam, sgd}
                --lr_decay_type <learning rate decay type> \       # default=multistep, options={multistep}
                --lr <initial learning rate>                       # default=1e-3
                --decay_epochs_train <epochs for multistep decay>  # default=30, options = epoch1 epoch2 ...
                --gamma <multiplicative factor of learning rate decay> \     # default=0.1
                --wd <weight decay>                                          # default=5e-4
                --seed <seed>                                                # default=0					
