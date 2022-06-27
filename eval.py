import torch
import numpy as np
from datetime import datetime
import pandas as pd

from utils import *
from dataloaders import get_pruning_examples, get_dataset, load_dataset
from train import training_loop

from lenet_prune import lenet_pruning
from vgg_prune import vgg_pruning
from resnet_prune import resnet_pruning

import argparse


def iterative_pruning(args, net, train_loader, test_loader, x_prune, device,
                      start_conv_prune=0, start_fc_prune=0):
    if 'lenet' in args.model_name:
        pruning = lenet_pruning
        total_params_mask = lenet_total_params_mask
        get_architecture = lenet_get_architecture
        compute_flops = lenet_compute_flops
    elif 'vgg' in args.model_name:
        pruning = vgg_pruning
        total_params_mask = vgg_total_params_mask
        get_architecture = vgg_get_architecture
        compute_flops = vgg_compute_flops
    else:
        pruning = resnet_pruning
        total_params_mask = resnet_total_params_mask
        get_architecture = resnet_get_architecture
        compute_flops = resnet_compute_flops

    cr = 1
    sparsity = 100

    columns = ['seed', 'iter', 'acc_pruned', 'acc_retrained', 'cr', 'remained_percentage', 'architecture', 'FLOPs']
    data = pd.DataFrame(columns=columns)
    acc = np.round(100 * accuracy(net, test_loader, device), 2)
    row = [args.seed, 0, acc, acc, cr, sparsity, get_architecture(net), compute_flops(net)]
    data.loc[0] = row

    if args.model_name == 'lenet300-100':
        x_prune = x_prune.reshape([-1, 28 * 28])

    init_masks_num = total_params_mask(net)
    init_flops = compute_flops(net)

    for it in range(1, args.num_iters + 1):
        # before_params_num = total_params(net)
        before_masks_num = total_params_mask(net)
        net.eval()
        net = pruning(net=net,
                      alpha_conv=args.alpha_conv,
                      alpha_fc=args.alpha_fc,
                      x_batch=x_prune,
                      device=device,
                      start_conv_prune=start_conv_prune,
                      start_fc_prune=start_fc_prune
                     )

        net._apply_mask()

        #after_params_num = total_params(net)
        after_masks_num = total_params_mask(net)
        curr_flops = compute_flops(net)
        acc_before = np.round(100 * accuracy(net, test_loader, device), 2)
        curr_arch = get_architecture(net)
        print('Accuracy before retraining: ', acc_before)
        print('Compression rate on iteration %i: ' %it, before_masks_num[0]/after_masks_num[0])
        #print('before, after: ', before_params_num, after_params_num)
        #print('mask before, mask after: ', before_masks_num, after_masks_num)
        print('Total compression rate: ', init_masks_num[0]/after_masks_num[0])
        print('The percentage of the remaining weights: ', 100*after_masks_num[0]/init_masks_num[0])
        print('FLOPs pruned: {}%'.format(np.round(100*(1-curr_flops/init_flops), 2)))
        print('Architecture: ', curr_arch)

        cr = np.round(init_masks_num[0]/after_masks_num[0], 2)
        sparsity = np.round(100*after_masks_num[0]/init_masks_num[0], 2)

        if 'lenet' in args.model_name:
            init_params = args.path_init_params
            net.load_state_dict(torch.load(init_params, map_location=device))
            net._apply_mask()

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.decay_epochs_retrain,
                                                        gamma=args.gamma)
        loss = torch.nn.CrossEntropyLoss()

        if args.model_name == 'lenet300-100':
            file_name = args.model_name+'_{}_{}_{}_{}_it{}_seed{}.pth'.format(args.optimizer, args.wd, 
                                                                                           args.dataset_name,
                                                                                           args.alpha_fc,
                                                                                           it,
                                                                                           args.seed)
        else:
            file_name = args.model_name+'_{}_{}_{}_{}_{}_it{}_seed{}.pth'.format(args.optimizer, args.wd, 
                                                                                              args.dataset_name,
                                                                                              args.alpha_conv,
                                                                                              args.alpha_fc,
                                                                                              it,
                                                                                              args.seed)                                                                                     

        net, _ = training_loop(model=net,
                               criterion=loss,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               train_loader=train_loader,
                               valid_loader=test_loader,
                               epochs=args.retrain_epochs,
                               model_name=args.model_name,
                               device=device,
                               file_name=file_name)

        net.load_state_dict(torch.load(file_name, map_location=device))
        
        acc_after = np.round(100 * accuracy(net, test_loader, device), 2)
        print('Accuracy after retraining: ', acc_after)

        row = [args.seed, it, acc_before, acc_after, cr, sparsity, curr_arch, curr_flops]
        data.loc[it] = row
        
        if args.model_name == 'lenet300-100':
            data.to_csv(args.model_name+'_{}_wd{}_{}_{}_seed{}.csv'.format(args.dataset_name, args.wd, args.optimizer, args.alpha_fc, args.seed), index=False)
        else:
            data.to_csv(args.model_name+'_{}_wd{}_{}_{}_{}_seed{}.csv'.format(args.dataset_name, args.wd, args.optimizer, args.alpha_conv, args.alpha_fc, args.seed), index=False)

        print('-------------------------------------------------')

    return net, data


def eval(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.dataset_name == 'mnist':
        train_dataset, test_dataset = get_dataset('mnist', download_data=args.download_data)
        num_classes = 10
    else:
        train_dataset, test_dataset = get_dataset(args.dataset_name, download_data=args.download_data)
        if args.dataset_name == 'cifar10':
            num_classes = 10
        else:
            num_classes = 100

    train_loader, test_loader = load_dataset(train_dataset, test_dataset, BATCH_SIZE=args.train_batch_size)

    x_prune = get_pruning_examples(dataset=train_dataset,
                                   device=device,
                                   dataset_name=args.dataset_name,
                                   prune_size=args.prune_batch_size,
                                   seed=args.seed)

    net = init_model(args.model_name, device, num_classes)
    pretrained_params = args.path_pretrained_model
    net.load_state_dict(torch.load(pretrained_params, map_location=device))

    print("EVALUATION...")

    acc = accuracy(net, test_loader, device)
    print('Accuracy: ', 100*np.round(acc, 4))
    
    net, stats = iterative_pruning(args=args,
                                   net=net,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   x_prune=x_prune,
                                   device=device)
                                   
                                   

    return net





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./', help='path to save/load dataset')
    parser.add_argument('--download_data', type=bool, default=True, help='download dataset')
    parser.add_argument('--model_name', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--path_pretrained_model', type=str, default='pretrained_model.pth', help='path to pretrained parameters')
    parser.add_argument('--path_init_params', type=str, default='init_params.pth', help='path to initialization parameters')
    parser.add_argument('--alpha_conv', type=float, default=0.9, help='fraction of importance to keep in conv layers')
    parser.add_argument('--alpha_fc', type=float, default=0.95, help='fraction of importance to keep in conv layers')
    parser.add_argument('--num_iters', type=int, default=5, help='number of pruning iterations')
    parser.add_argument('--prune_batch_size', type=int, default=1000, help='number of examples for pruning')
    parser.add_argument('--train_batch_size', type=int, default=120, help='number of examples per training batch')
    parser.add_argument('--retrain_epochs', type=int, default=60, help='number of retraining epochs after pruning')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='multistep', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--decay_epochs_retrain', nargs='+', type=int, default=[30], help='epochs for multistep decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay during retraining')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    args = parser.parse_args()

    eval(args)

