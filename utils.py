import torch
import numpy as np
from datetime import datetime

from models import LeNet5_Caffe, LeNet300_100, VGG, resnet20, resnet32, resnet44, resnet56, resnet110


def init_model(model_name, device, num_classes=10):
    if 'lenet' in model_name:
          if 'lenet5' == model_name:
              model = LeNet5_Caffe(num_classes=num_classes, device=device)
          else:
              model = LeNet300_100(num_classes=num_classes, device=device)
    elif 'vgg' in model_name:
        model = VGG(model_name, num_classes=num_classes, device=device)

    elif model_name == 'resnet20':
        model = resnet20(num_classes, device);
    elif model_name == 'resnet32':
        model = resnet32(num_classes, device);
    elif model_name == 'resnet44':
        model = resnet44(num_classes, device);
    elif model_name == 'resnet56':
        model = resnet56(num_classes, device);
    elif model_name == 'resnet110':
        model = resnet110(num_classes, device);

    model = model.to(device)

    return model


def accuracy(model, data_loader, device):
    correct_preds = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_preds = model(X)

            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()

    return (correct_preds / n).item()

def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def lenet_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def lenet_total_params_mask(model):
    total_number_conv = torch.tensor(0, dtype=torch.int32)
    if list(model.named_children())[0][0] == 'feature_extractor':
        for mask in model.feature_extractor_masks:
            total_number_conv += mask.sum().int()

    total_number_fc = torch.tensor(0, dtype=torch.int32)
    for mask in model.classifier_masks:
        total_number_fc += mask.sum().int()

    total_number = total_number_conv + total_number_fc


    return total_number.item(), total_number_conv.item(), total_number_fc.item()


def lenet_get_architecture(model):
    arch = []
    convs = []
    fc = []

    if list(model.named_children())[0][0] == 'feature_extractor':
        for h in range(len(model.feature_extractor_masks) // 2):
            convs.append(torch.sum(model.feature_extractor_masks[2*h].sum(dim=(1, 2, 3)) > 0).item())

    arch.append(convs)

    for h in range(len(model.classifier_masks)//2):
        fc.append(torch.sum(model.classifier_masks[2 * h].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def lenet_compute_flops(model):
    arch = lenet_get_architecture(model)
    flops = 0

    h = [24, 8]
    k = 5
    channels = [1] + arch[0]
    for l in range(len(channels) - 1):
        flops += 2*h[l]*h[l]*(channels[l] * k**2 + 1)*channels[l+1]

    for i in range(len(arch[1])-1):
        flops += (2*arch[1][i]-1)*arch[1][i+1]

    flops += (2*arch[1][-1]-1)*10

    return flops


def vgg_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def vgg_total_params_mask(model):
    total_number_conv = 0
    if list(model.named_children())[0][0] == 'features':
        for mask in model.features_masks:
            total_number_conv += mask.sum()
            # print(torch.numel(mask[mask != 0]))

    total_number_fc = 0
    for mask in model.classifier_masks:
        total_number_fc += mask.sum()
        # print(torch.numel(mask[mask != 0]))

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def vgg_get_architecture(model):
    arch = []
    convs = []
    fc = []

    if list(model.named_children())[0][0] == 'features':
        for h in range(len(model.features_masks)//2):
            convs.append(torch.sum(model.features_masks[2*h].sum(dim=(1, 2, 3)) > 0).item())

    arch.append(convs)

    for h in range(len(model.classifier_masks)//2):
        fc.append(torch.sum(model.classifier_masks[2*h].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def vgg_compute_flops(model):
    arch = vgg_get_architecture(model)
    flops = 0
    h = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    k = 3
    channels = [3] + arch[0]

    for l in range(len(channels) - 1):
        flops += 2*h[l]*h[l]*(channels[l]*k**2 + 1)*channels[l+1]

    for i in range(len(arch[1])-1):
        flops += (2*arch[1][i]-1)*arch[1][i+1]

    flops += (2*arch[1][-1]-1)*10

    return flops


def resnet_total_params(model):
    total_number = 0
    for param_name in list(model.state_dict()):
        param = model.state_dict()[param_name]
        total_number += torch.numel(param[param != 0])

    return total_number


def resnet_total_params_mask(model):
    total_number_conv = 0
    total_number_fc = 0

    for name, param in list(model.named_parameters()):
        if (name == 'conv1'):
            total_number_conv += model.conv1_masks.sum()
        elif ('linear' in name):
            if ('weight' in name):
                total_number_fc += model.linear_masks[0].sum()
            else:
                total_number_fc += model.linear_masks[1].sum()
        else:
            for layer in range(len(model.num_blocks)):
                for block in range(model.num_blocks[layer]):
                    if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                        total_number_conv += model.layers_masks[layer][block][0].sum()
                    elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                        total_number_conv += model.layers_masks[layer][block][1].sum()

    total = total_number_conv + total_number_fc

    return total.item(), total_number_conv.item(), total_number_fc.item()


def resnet_get_architecture(model):
    arch = []
    convs = []
    fc = []

    convs.append(torch.sum(model.conv1_masks.sum(dim=(1, 2, 3)) > 0).item())
    for block, num_block in enumerate(model.num_blocks):
        block_masks = []
        for i in range(num_block):
            block_conv_masks = []
            for conv_num in range(2):
                block_conv_masks.append(torch.sum(model.layers_masks[block][i][conv_num].sum(dim=(1, 2, 3)) > 0).item())

            block_masks.append(block_conv_masks)

        convs.append(block_masks)

    arch.append(convs)

    fc.append(torch.sum(model.linear_masks[0].sum(dim=0) > 0).item())

    arch.append(fc)

    return arch


def resnet_compute_flops(model):
    arch_conv, arch_fc = resnet_get_architecture(model)
    flops = 0
    k = 3
    h = 32
    w = 32
    in_channels = 3

    flops += (2 * h * w * (in_channels * k ** 2) - 1) * arch_conv[0]

    in_channels = arch_conv[0]

    for block, num_block in enumerate(model.num_blocks):
        for i in range(num_block):
            for conv_num in range(2):
                if (conv_num == 1):
                    out_channels = torch.sum(torch.logical_or(model.layers_masks[block][i][1].sum(dim=(1, 2, 3)),
                                                              model.layers_masks[block][i][-1]) > 0).item()
                else:
                    out_channels = arch_conv[1 + block][i][conv_num]

                flops += (2 * h * w * (in_channels * k ** 2) - 1) * out_channels

            in_channels = out_channels

            flops += h * w * model.layers_masks[block][i][-1].sum().item()

        h /= 2
        w /= 2

    flops += (2 * arch_fc[0] - 1) * 10

    return flops

