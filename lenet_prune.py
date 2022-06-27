import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def lenet_fc_pruning(net, alpha, x_batch, device, start_prune=0):
    layers = list(net.classifier.state_dict())

    num_samples = x_batch.size()[0]

    for l in range(start_prune, len(layers)//2):
        curr_layer = list(net.classifier.named_children())[2*l][1](x_batch)

        fc_weight = net.classifier.state_dict()[layers[2*l]]
        fc_bias = net.classifier.state_dict()[layers[2*l+1]]

        weights_mask = torch.ones(fc_weight.shape).to(device)
        bias_mask = torch.ones(fc_bias.shape).to(device)

        if 2*l+1 < len(list(net.classifier.named_children())):
            activation = lambda x: torch.nn.functional.relu(x)
        else:
            activation = lambda x: x

        for i in range(curr_layer.size()[1]):
            avg_neuron_val = torch.mean(activation(curr_layer), axis=0)[i]

            if (avg_neuron_val == 0):
                weights_mask[i] = 0
                bias_mask[i] = 0
            else:
                flow = torch.cat((x_batch*fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1).abs()

                importances = torch.mean(torch.abs(flow), dim=0)

                sum_importance = torch.sum(importances)
                sorted_importances, sorted_indices = torch.sort(importances, descending=True)

                cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
                pivot = torch.sum(cumsum_importances < alpha*sum_importance)

                if pivot < importances.size(0) - 1:
                    pivot += 1
                else:
                    pivot = importances.size(0) - 1

                thresh = importances[sorted_indices][pivot]

                fc_weight[i][importances[:-1] <= thresh] = 0
                weights_mask[i][importances[:-1] <= thresh] = 0

                if importances[-1] <= thresh:
                    bias_mask[i] = 0

        net.classifier_masks[2*l] = weights_mask.cpu()
        net.classifier_masks[2*l+1] = bias_mask.cpu()
        
        x_batch = activation(curr_layer)

    net._apply_mask()

    return net


def lenet_conv_pruning(net, alpha, x_batch, device,  start_prune = 0):
    layers = list(net.feature_extractor.named_children())

    activation = lambda x: torch.nn.functional.relu(x)

    for l in range(start_prune):
        x_batch = layers[3*l][1](x_batch)    # CONV
        x_batch = layers[3*l+1][1](x_batch)  # activation
        x_batch = layers[3*l+2][1](x_batch)  # max-pool

    for l in range(start_prune, len(layers )//3):
        # pruned_conv = 0

        conv_out = activation( layers[3*l][1](x_batch) )
        avg_maps_val = torch.mean(conv_out, dim=0)

        bias = list(net.feature_extractor.named_parameters())[2*l+1][1].data.cpu()

        padding = layers[3*l][1].padding
        stride = layers[3*l][1].stride
        kernel_size = layers[3*l][1].kernel_size

        p2d = (padding[0],) * 2 + (padding[1], )* 2

        n = x_batch.size(3)
        m = x_batch.size(2)

        zero_kernel = torch.zeros(kernel_size)

        kernels = list(net.feature_extractor.named_parameters())[2*l][1].data

        x_batch = F.pad(x_batch,
                        p2d,
                        "constant",
                        0)

        importances = torch.zeros(kernels.size(0), kernels.size(1),
                                  (n+2*padding[0])-2*(kernel_size[0]//2),
                                  (m+2*padding[1])-2*(kernel_size[1]//2))

        for i in range(kernel_size[0]//2, (n+2*padding[0])-kernel_size[0]//2, stride[0]):
            for j in range(kernel_size[1]//2, (m + 2 * padding[1]) - kernel_size[1] // 2, stride[1]):
                inp = x_batch[:, :, (i-kernel_size[0]//2):(i + kernel_size[0]//2+1),
                                (j-kernel_size[1]//2):(j+kernel_size[1]//2+1)]

                inp = inp.unsqueeze(1).expand(-1, kernels.size(0), -1, -1, -1)

                importances[:, :, i-kernel_size[0]//2, j-kernel_size[1]//2] = (inp[:].abs()*kernels.abs()).sum(dim=(3, 4)).mean(dim=0)

        bias = (bias.reshape(-1, 1, 1, 1)).expand(-1, -1, importances.size(-2), importances.size(-1))

        importances = torch.cat((importances, bias.abs()), dim=1)

        importances = torch.sum(importances, dim=(2, 3))
        sorted_importances, sorted_indices = torch.sort(importances, dim=1, descending=True)

        for k in range(importances.size(0)):
            if torch.sum(avg_maps_val[k] > 0) == 0:
                net.feature_extractor_masks[2*l][k] = zero_kernel
                net.feature_extractor_masks[2*l+1][k] = 0
                #pruned_conv += net.feature_extractor_masks[2*l][k].numel()+net.feature_extractor_masks[2*l+1][k].numel()
            else:
                importance_cumsum = torch.cumsum(importances[k, sorted_indices[k]], dim=0)
                importance_sum = torch.sum(importances[k], dim=0).unsqueeze(0)

                pivot = torch.sum(importance_cumsum < alpha * importance_sum)

                if pivot < importances[k].size(0) - 1:
                    pivot += 1
                else:
                    pivot = importances[k].size(0) - 1

                # delete all connectons that are less important than the pivot
                thresh = sorted_importances[k][pivot]

                kernel_zero_idx = torch.nonzero(importances[k][:-1] <= thresh)
                is_bias_zero = int(importances[k][-1] <= thresh)

                net.feature_extractor_masks[2*l][k][kernel_zero_idx] = zero_kernel
                net.feature_extractor_masks[2*l+1][k] = 1 - is_bias_zero

                #pruned_conv += is_bias_zero+kernel_zero_idx.size(0)*kernel_size[0]*kernel_size[1]

        net._apply_mask()

        x_batch = conv_out
        x_batch = layers[3*l+2][1](x_batch)

        # compute accuracy and the number of non-zero parameters
        # acc = accuracy(net, test_loader, device)
        # print("Accuracy after pruning the conv layer %d: " %l, acc)
        # print("The percentage of pruned parameters in the conv layer %d: " % l,
        #      pruned_conv/(torch.numel(net.feature_extractor_masks[2*l]) +
        #                     torch.numel(net.feature_extractor_masks[2*l+1])))
        # print("#################################")


    return net


def lenet_backward_pruning(net):
    h = len(net.classifier_masks )//2-1
    while h > 0:
        pruned_neurons = torch.nonzero( net.classifier_masks[2*h].sum(dim=0) == 0).reshape(1, -1).squeeze(0)
        net.classifier_masks[2*(h-1)][pruned_neurons] = 0
        net.classifier_masks[2*(h-1)+1][pruned_neurons] = 0

        h -= 1

    if list(net.named_children())[0][0] == 'feature_extractor':
        l = len(list(net.feature_extractor.named_children()))-3

        kernel_size = list(net.feature_extractor.named_children())[l][1].kernel_size
        zero_kernel = torch.zeros(kernel_size)
        pruned_channels = torch.nonzero( net.classifier_masks[0].sum(dim=0).reshape(50,4,4).sum(dim=(1,2)) == 0).reshape(1, -1).squeeze(0)

        if len(pruned_channels) > 0:
            net.feature_extractor_masks[-2][pruned_channels] = zero_kernel
            net.feature_extractor_masks[-1][pruned_channels] = 0

        h = len(net.feature_extractor_masks )//2-1

        l -= 3

        while l >= 0:
            kernel_size = list(net.feature_extractor.named_children())[l][1].kernel_size
            zero_kernel = torch.zeros(kernel_size)

            pruned_channels = torch.nonzero(net.feature_extractor_masks[2*h].sum(dim=(0 ,2 ,3))==0).reshape(1, -1).squeeze(0)

            if len(pruned_channels) > 0:
                # print(pruned_channels)
                if net.feature_extractor_masks[2*(h-1)].size(1) > 1:
                    net.feature_extractor_masks[2*(h-1)][pruned_channels] = zero_kernel
                else:
                    net.feature_extractor_masks[2*(h-1)][pruned_channels] = zero_kernel

                net.feature_extractor_masks[2*(h-1)+1][pruned_channels] = 0

            h -= 1
            l -= 3

    net._apply_mask()

    return net


def lenet_pruning(net, alpha_conv, alpha_fc, x_batch, device,
                  start_conv_prune=0, start_fc_prune=0):

    if (list(net.named_children())[0][0] == 'feature_extractor'):
        if (start_conv_prune >= 0):
            net = lenet_conv_pruning(net, alpha_conv, x_batch, device, start_conv_prune)
        x_batch = net.feature_extractor(x_batch)
        x_batch = x_batch.view(x_batch.size(0), x_batch.size(1) * x_batch.size(2) * x_batch.size(3))

    # pruner for fully-connected layers
    if (start_fc_prune >= 0):
        net = lenet_fc_pruning(net, alpha_fc, x_batch, device, start_fc_prune)

    #print('---Before backward: ', total_params_mask(net))
    net = lenet_backward_pruning(net)
    #print('---After backward: ', total_params_mask(net))
    return net
