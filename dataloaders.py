import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
import numpy as np


def get_pruning_examples(dataset, device, dataset_name='mnist', prune_size=1000, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    prune_idx = np.random.permutation(dataset.data.shape[0])[:prune_size]

    if dataset_name == 'mnist':
        x_prune = dataset.data[prune_idx].float().to(device)
        x_prune = x_prune.unsqueeze(1).float()/255
        #np.savetxt('prune_idx_mnist_seed{}.csv'.format(seed), prune_idx)
    else:
        x_prune = torch.FloatTensor(dataset.data[prune_idx]).to(device)
        x_prune = x_prune.permute(0, 3, 1, 2)
        x_prune = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x_prune/255)

    #np.savetxt('prune_idx_{}_seed{}.csv'.format(dataset_name, seed), prune_idx)

    return x_prune


def get_dataset(dataset_name, download_data=True):
    if dataset_name == 'mnist':
        transform = transforms.Compose([torchvision.transforms.ToTensor()])

        train_dataset = torchvision.datasets.MNIST(root='./',
                                                   train=True,
                                                   transform=transform,
                                                   download=download_data)
        test_dataset = torchvision.datasets.MNIST(root='./',
                                                  train=False,
                                                  transform=transform,
                                                  download=download_data)
    else:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                              ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])
        if dataset_name == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root='./',
                                                         train=True,
                                                         transform=transform_train,
                                                         download=download_data)
            test_dataset = torchvision.datasets.CIFAR10(root='./',
                                                        train=False,
                                                        transform=transform_test,
                                                        download=download_data)
        else:
            train_dataset = torchvision.datasets.CIFAR100(root='./',
                                                          train=True,
                                                          transform=transform_train,
                                                          download=download_data)
            test_dataset = torchvision.datasets.CIFAR100(root='./',
                                                         train=False,
                                                         transform=transform_test,
                                                         download=download_data)
    return train_dataset, test_dataset


def load_dataset(train_dataset, test_dataset, BATCH_SIZE=120):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=4)

    return train_loader, test_loader
