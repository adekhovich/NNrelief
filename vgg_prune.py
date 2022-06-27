import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def vgg_fc_pruning(net, alpha, x_batch, device, start_prune=0):
    layers = list(net.classifier.state_dict())

    num_samples = x_batch.size()[0]

    for l in range(start_prune, len(layers) // 2):

        fc_weight = net.classifier.state_dict()[layers[2 * l]]
        fc_bias = net.classifier.state_dict()[layers[2 * l + 1]]

        weights_mask = torch.ones(fc_weight.shape).to(device)
        bias_mask = torch.ones(fc_bias.shape).to(device)

        if 3*l + 1 < len(list(net.classifier.named_children())):
            activation = lambda x: torch.nn.functional.relu(x)
            curr_layer = list(net.classifier.named_children())[3*l][1](x_batch)
            curr_layer = list(net.classifier.named_children())[3*l+1][1](curr_layer)
        else:
            activation = lambda x: x
            curr_layer = list(net.classifier.named_children())[3*l][1](x_batch)

        for i in range(curr_layer.size()[1]):
            avg_neuron_val = torch.mean(activation(curr_layer), axis=0)[i]

            if (avg_neuron_val == 0):
                weights_mask[i] = 0
                bias_mask[i] = 0

            else:
                flow = torch.cat((x_batch * fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1)
                importances = torch.mean(torch.abs(flow), dim=0)

                sum_importance = torch.sum(importances)
                sorted_importances, sorted_indices = torch.sort(importances, descending=True)

                cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
                pivot = torch.sum(cumsum_importances < alpha*sum_importance)

                if pivot < importances.size(0)-1:
                    pivot += 1
                else:
                    pivot = importances.size(0)-1

                thresh = importances[sorted_indices][pivot]

                weights_mask[i][importances[:-1] <= thresh] = 0

                if (importances[-1] <= thresh):
                    bias_mask[i] = 0

        net.classifier_masks[2*l] = weights_mask.cpu()
        net.classifier_masks[2*l+1] = bias_mask.cpu()

        net._apply_mask()
        x_batch = activation(curr_layer)

    return net


def vgg_conv_pruning(net, alpha, x_batch, device, start_prune=0):
    net.eval()
    layers = list(net.features.named_children())
    # params = list(net.features.named_parameters())

    step = 4  # if our model has batchnorm layers, otherwise step=2
    l_list = []
    l = start_prune
    h = 0
    c = 0

    while (l < len(layers) - 3):
        if (net.cfg[c] == 'M'):
            x_batch = list(net.features.named_children())[l][1](x_batch)
            l += 1
            c += 1

        # pruned_conv = 0
        # compute filter output
        block_out = list(net.features.named_children())[l][1](x_batch)  # conv
        block_out = list(net.features.named_children())[l + 1][1](block_out)  # batchnorm
        block_out = list(net.features.named_children())[l + 2][1](block_out)  # activation
        avg_maps_val = torch.mean(block_out, dim=0)  # averaging

        bias = list(net.features.named_parameters())[step*h+1][1].data.cpu()

        padding = list(net.features.named_children())[l][1].padding
        stride = list(net.features.named_children())[l][1].stride
        kernel_size = list(net.features.named_children())[l][1].kernel_size

        p2d = (padding[0],) * 2 + (padding[1],) * 2
        n = x_batch.size(3)
        m = x_batch.size(2)

        zero_kernel = torch.zeros(kernel_size)
        kernels = list(net.features.named_parameters())[step*h][1].data

        x_batch = F.pad(x_batch, p2d, "constant", 0)

        for k in range(kernels.size(0)):
            importances = torch.zeros(kernels.size(1), (n+2*padding[0])-2*(kernel_size[0]//2), (m+2*padding[1])-2*(kernel_size[1]//2))

            for i in range(kernel_size[0]//2, (n+2*padding[0])-kernel_size[0]//2, stride[0]):
                for j in range(kernel_size[1]//2, (m+2*padding[1])-kernel_size[1]//2, stride[1]):
                    input = x_batch[:, :, (i-kernel_size[0]//2):(i+kernel_size[0]//2+1),
                            (j-kernel_size[1]//2):(j+kernel_size[1]//2+1)].abs().mean(dim=0)

                    importances[:, i-kernel_size[0]//2, j-kernel_size[1]//2] = torch.sum(torch.abs(input*kernels[k]), dim=(1, 2))

            if avg_maps_val[k].norm(dim=(0, 1)) == 0:
                net.features_masks[2*h][k] = zero_kernel
                net.features_masks[2*h+1][k] = 0
                # batchnorm2d params

                net.features.state_dict()[str(l + 1) + '.weight'][k] = 0
                net.features.state_dict()[str(l + 1) + '.bias'][k] = 0
                net.features.state_dict()[str(l + 1) + '.running_mean'][k] = 0
                net.features.state_dict()[str(l + 1) + '.running_var'][k] = 0
                #pruned_conv += net.features_masks[2 * h][k].numel()+net.features_masks[2*h+1][k].numel()
            else:
                bias_mat = (bias[k].reshape(-1, 1, 1)).expand(-1, importances.size(-2), importances.size(-1))
                importances = torch.cat((importances, bias_mat.abs()), dim=0)
                importances = torch.norm(importances, dim=(1, 2))

                sorted_importances, sorted_indices = torch.sort(importances, dim=0, descending=True)
                pivot = torch.sum(sorted_importances.cumsum(dim=0) < alpha*importances.sum())

                if pivot < importances.size(0) - 1:
                    pivot += 1
                else:
                    pivot = importances.size(0) - 1

                # delete all connectons that are less important than the pivot
                thresh = sorted_importances[pivot]

                kernel_zero_idx = torch.nonzero(importances[:-1] <= thresh).reshape(1, -1).squeeze(0)
                is_bias_zero = 1*(importances[-1] <= thresh)

                net.features_masks[2*h][k][kernel_zero_idx] = zero_kernel
                net.features_masks[2*h+1][k] = 1-is_bias_zero
                # pruned_conv += is_bias_zero+kernel_zero_idx.size(0)*kernel_size[0]*kernel_size[1]

        net._apply_mask()

        x_batch = block_out

        h += 1
        c += 1
        l_list.append(l)
        l += 3

    return net


def vgg_backward_pruning(net):
    h = len(net.classifier_masks)//2-1
    while h > 0:
        pruned_neurons = torch.nonzero(net.classifier_masks[2*h].sum(dim=0) == 0).reshape(1, -1).squeeze(0)
        net.classifier_masks[2*(h-1)][pruned_neurons] = 0
        net.classifier_masks[2*(h-1)+1][pruned_neurons] = 0
        h -= 1

    l = len(list(net.features.named_children()))-1-3
    c = len(net.cfg)-2

    kernel_size = list(net.features.named_children())[l][1].kernel_size
    zero_kernel = torch.zeros(kernel_size)
    pruned_channels = torch.nonzero(net.classifier_masks[0].sum(dim=0) == 0).reshape(1, -1).squeeze(0)
    if len(pruned_channels) > 0:
        net.features_masks[-2][pruned_channels] = zero_kernel
        net.features_masks[-1][pruned_channels] = 0

        net.features.state_dict()[str(l + 1) + '.weight'][pruned_channels] = 0
        net.features.state_dict()[str(l + 1) + '.bias'][pruned_channels] = 0
        net.features.state_dict()[str(l + 1) + '.running_mean'][pruned_channels] = 0
        net.features.state_dict()[str(l + 1) + '.running_var'][pruned_channels] = 0

    h = len(net.features_masks)//2-1
    c -= 1

    if net.cfg[c] == 'M':
        l -= 1
    else:
        l -= 3

    while l >= 0:
        if net.cfg[c] == 'M':
            l -= 3
            c -= 1

        kernel_size = list(net.features.named_children())[l][1].kernel_size
        zero_kernel = torch.zeros(kernel_size)
        pruned_channels = torch.nonzero(net.features_masks[2*h].sum(dim=(0, 2, 3)) == 0).reshape(1, -1).squeeze(0)

        if len(pruned_channels) > 0:
            net.features_masks[2*(h-1)][pruned_channels] = zero_kernel
            net.features_masks[2*(h-1)+1][pruned_channels] = 0

            net.features.state_dict()[str(l+1) + '.weight'][pruned_channels] = 0
            net.features.state_dict()[str(l+1) + '.bias'][pruned_channels] = 0
            net.features.state_dict()[str(l+1) + '.running_mean'][pruned_channels] = 0
            net.features.state_dict()[str(l+1) + '.running_var'][pruned_channels] = 0

        h -= 1
        c -= 1

        if net.cfg[c] == 'M':
            l -= 1
        else:
            l -= 3

    return net


def vgg_pruning(net, alpha_conv, alpha_fc, x_batch, device, start_conv_prune=0, start_fc_prune=0):
    if list(net.named_children())[0][0] == 'features':
        if start_conv_prune >= 0:
            net = vgg_conv_pruning(net, alpha_conv, x_batch, device, start_conv_prune)

    x_batch = net.features(x_batch)
    x_batch = x_batch.view(x_batch.size(0), -1)

    if start_fc_prune >= 0:
        net = vgg_fc_pruning(net, alpha_fc, x_batch, device, start_fc_prune)

    net = vgg_backward_pruning(net)

    return net
