import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def resnet_fc_pruning(net, alpha, x_batch, start_fc_prune, device):
    layers = list(net.linear.state_dict())
    num_samples = x_batch.size()[0]

    fc_weight = net.linear.state_dict()[layers[0]]
    fc_bias = net.linear.state_dict()[layers[1]]

    weights_mask = torch.ones(fc_weight.shape).to(device)
    bias_mask = torch.ones(fc_bias.shape).to(device)

    curr_layer = net.linear(x_batch)

    for i in range(curr_layer.size(1)):
        flow = torch.cat((x_batch * fc_weight[i], torch.reshape(fc_bias[i].repeat(num_samples), (-1, 1))), dim=1)
        importances = torch.mean(torch.abs(flow), dim=0)

        sum_importance = torch.sum(importances)
        sorted_importances, sorted_indices = torch.sort(importances, descending=True)

        cumsum_importances = torch.cumsum(importances[sorted_indices], dim=0)
        pivot = torch.sum(cumsum_importances < alpha * sum_importance)

        if pivot < importances.size(0) - 1:
            pivot += 1
        else:
            pivot = importances.size(0) - 1

        thresh = importances[sorted_indices][pivot]

        fc_weight[i][importances[:-1] <= thresh] = 0
        weights_mask[i][importances[:-1] <= thresh] = 0

        if importances[-1] <= thresh:
            fc_bias[i] = 0
            bias_mask[i] = 0

    # set new parameters
    net.linear.state_dict()[layers[0]] = fc_weight
    net.linear.state_dict()[layers[1]] = fc_bias

    net.linear_masks[0] = weights_mask.cpu()
    net.linear_masks[1] = bias_mask.cpu()

    return net


def resnet_conv_block_pruning(net, layer_num, block_num, conv_num, alpha, x_batch, residual=0):
    if layer_num == 0:
        conv = net.conv1
        bn = net.bn1
        name = 'conv1.weight'
        name_bn = 'bn1'
    else:
        conv = list(list(net.named_children())[layer_num + 1][1][block_num].named_children())[2 * conv_num][1]
        bn = list(list(net.named_children())[layer_num + 1][1][block_num].named_children())[2 * conv_num + 1][1]
        name = 'layer{}.{}.conv{}.weight'.format(layer_num, block_num, conv_num + 1)
        name_bn = 'layer{}.{}.bn{}'.format(layer_num, block_num, conv_num + 1)

    bn_out = bn(conv(x_batch))
    if conv_num == 1:
        block_out = F.relu(bn_out + residual)
    else:
        block_out = F.relu(bn_out)

    block_out_mean = block_out.mean(dim=0)

    padding = conv.padding
    stride = conv.stride
    kernel_size = conv.kernel_size
    zero_kernel = torch.zeros(kernel_size)

    filters = net.state_dict()[name]

    p2d = (padding[0],) * 2 + (padding[1],) * 2
    n = x_batch.size(3)
    m = x_batch.size(2)

    x_batch = F.pad(x_batch, p2d, "constant", 0)

    for k in range(filters.size(0)):

        if (block_out_mean[k]).norm(dim=(0, 1)) == 0:
            if layer_num == 0:
                net.conv1_masks[k] = zero_kernel
            else:
                net.layers_masks[layer_num - 1][block_num][conv_num][k] = zero_kernel
                if conv_num == 1:
                    net.layers_masks[layer_num - 1][block_num][-1][k] = 0

            net.state_dict()[name_bn + '.weight'][k] = 0
            net.state_dict()[name_bn + '.bias'][k] = 0
            net.state_dict()[name_bn + '.running_mean'][k] = 0
            net.state_dict()[name_bn + '.running_var'][k] = 0
        else:
            importances = torch.zeros(filters.size(1), ((n+2* padding[0] - kernel_size[0])//stride[0]+1), ((m+2*padding[1]-kernel_size[1])//stride[1]+1))

            for i in range(kernel_size[0]//2, (n+2*padding[0])-kernel_size[0]//2, stride[0]):
                for j in range(kernel_size[1]//2, (m+2*padding[1])-kernel_size[1]//2, stride[1]):
                    input = x_batch[:, :, (i-kernel_size[0]//2):(i+kernel_size[0]//2+1),
                            (j-kernel_size[1]//2):(j+kernel_size[1]//2+1)].abs().mean(dim=0)

                    importances[:, (i-kernel_size[0]//2)//stride[0], (j-kernel_size[1]//2)//stride[1]] = torch.sum(torch.abs(input*filters[k]), dim=(1, 2))

            importances = torch.norm(importances, dim=(1, 2))
            sorted_importances, sorted_indices = torch.sort(importances, dim=0, descending=True)

            pivot = torch.sum(sorted_importances.cumsum(dim=0) < alpha*importances.sum())
            if pivot < importances.size(0) - 1:
                pivot += 1
            else:
                pivot = importances.size(0) - 1

            # delete all connectons that are less important than the pivot
            thresh = sorted_importances[pivot]
            kernel_zero_idx = torch.nonzero(importances <= thresh).reshape(1, -1).squeeze(0)

            if layer_num == 0:
                net.conv1_masks[k][kernel_zero_idx] = zero_kernel
            else:
                net.layers_masks[layer_num - 1][block_num][conv_num][k][kernel_zero_idx] = zero_kernel

    if conv_num == 1:
        pruned_channels = torch.nonzero(
            bn_out.abs().mean(dim=0).norm(dim=(1, 2))/residual.abs().mean(dim=0).norm(dim=(1, 2)) < (1-alpha)/alpha).reshape(1, -1).squeeze(0)
        net.layers_masks[layer_num - 1][block_num][conv_num][pruned_channels] = zero_kernel

        pruned_channels = torch.nonzero(
            residual.abs().mean(dim=0).norm(dim=(1, 2))/bn_out.abs().mean(dim=0).norm(dim=(1, 2)) < (1-alpha)/alpha).reshape(1, -1).squeeze(0)
        net.layers_masks[layer_num - 1][block_num][-1][pruned_channels] = 0

    net._apply_mask()

    return net, block_out


def resnet_conv_pruning(net, alpha, x_batch, start_conv_prune, device):
    net.eval()
    named_params = list(net.named_parameters())

    for name, param in named_params:
        if (name == 'conv1.weight'):
            net, x_batch = resnet_conv_block_pruning(net, 0, 0, 0, alpha, x_batch)
        else:
            for layer in range(len(net.num_blocks)):
                for block in range(net.num_blocks[layer]):
                    for conv_num in range(2):
                        if (name == 'layer{}.{}.conv{}.weight'.format(layer+1, block, conv_num+1)):
                            if (conv_num == 0):
                                residual = list(list(net.named_children())[layer+2][1][block].named_children())[-1][1](x_batch)
                            net, x_batch = resnet_conv_block_pruning(net, layer+1, block, conv_num, alpha, x_batch, residual)
    return net


def resnet_backward_pruning(net):
    pruned_channels = torch.nonzero(net.linear_masks[0].sum(dim=0) == 0).reshape(1, -1).squeeze(0)

    kernel_size = list(list(net.layer3.named_children())[-1][1].named_children())[-3][1].kernel_size
    zero_kernel = torch.zeros(kernel_size)

    for name in reversed(list(net.state_dict())):
        for layer in range(len(net.num_blocks))[::-1]:
            for block in range(net.num_blocks[layer])[::-1]:
                basic_block = list(net.children())[layer+2][block]

                if basic_block.planes != basic_block.in_planes:
                    start = basic_block.planes//4
                    end = basic_block.planes - basic_block.planes//4
                else:
                    start = 0
                    end = basic_block.planes

                for conv_num in range(2)[::-1]:
                    if (('layer{}.{}.bn{}'.format(layer+1, block, conv_num+1) in name) and
                            ('num_batches_tracked' not in name)):
                        net.state_dict()[name][pruned_channels] = 0
                    elif (name == 'layer{}.{}.conv{}.weight'.format(layer+1, block, conv_num+1)):
                        net.layers_masks[layer][block][conv_num][pruned_channels] = zero_kernel

                        if (conv_num == 1):
                            net.layers_masks[layer][block][-1][pruned_channels] = 0
                            pruned_channels = torch.nonzero(
                                net.layers_masks[layer][block][conv_num].sum(dim=(0, 2, 3)) == 0).reshape(1,-1).squeeze(0)
                        else:
                            pruned_channels = torch.nonzero((1 - net.layers_masks[layer][block][-1][start:end])*(
                                        net.layers_masks[layer][block][conv_num].sum(dim=(0, 2, 3)) == 0)).reshape(1, -1).squeeze(0)

    net.conv1_masks[pruned_channels] = zero_kernel

    return net


def resnet_pruning(net, alpha_conv, alpha_fc, x_batch, device, 
                   start_conv_prune=0, start_fc_prune=0):
    if (start_conv_prune >= 0):
        net = resnet_conv_pruning(net, alpha_conv, x_batch, start_conv_prune, device)

    x_batch = net.features(x_batch)

    if (start_fc_prune >= 0):
        net = resnet_fc_pruning(net, alpha_fc, x_batch, start_fc_prune, device)

    net = resnet_backward_pruning(net)
    return net