import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class LeNet300_100(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(LeNet300_100, self).__init__()

        self.num_classes = num_classes
        self.device = device
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=784, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=10),
        )

        self.apply(_weights_init)

        self.classifier_masks = [
                                 torch.ones(300, 784),
                                 torch.ones(300),
                                 torch.ones(100, 300),
                                 torch.ones(100),
                                 torch.ones(10, 100),
                                 torch.ones(10)
        ]

    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.classifier(x)
        
        return x

    def _apply_mask(self):
        if (list(self.named_children())[0][0] == 'feature_extractor'):
            l = 0
            for name, param in self.feature_extractor.named_parameters():
                param.data = param.data*(self.feature_extractor_masks[l]).to(self.device)
                l += 1
            
        l = 0
        for name, param in self.classifier.named_parameters():
            param.data = param.data*(self.classifier_masks[l]).to(self.device)
            l += 1  


class LeNet5_Caffe(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(LeNet5_Caffe, self).__init__()

        self.num_classes = num_classes
        self.device = device
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.feature_extractor_masks = [
                                        torch.ones(20, 1, 5, 5),
                                        torch.ones(20),
                                        torch.ones(50, 20, 5, 5),
                                        torch.ones(50)
        ]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*50, out_features=500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=500, out_features=10),
        )

        self.classifier_masks = [
                                 torch.ones(500, 4*4*50),
                                 torch.ones(500),
                                 torch.ones(10, 500),
                                 torch.ones(10)
        ]

        self.apply(_weights_init)

    
    def forward(self, x):
        
        x = self.feature_extractor(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.classifier(x)
        
        return x

    def _apply_mask(self):
        if (list(self.named_children())[0][0] == 'feature_extractor'):
            l = 0
            for name, param in self.feature_extractor.named_parameters():
                param.data = param.data*(self.feature_extractor_masks[l]).to(self.device)
                l += 1
            
        l = 0
        for name, param in self.classifier.named_parameters():
            param.data = param.data*(self.classifier_masks[l]).to(self.device)
            l += 1       



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG-like': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg-like': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, device):
        super(VGG, self).__init__()

        self.cfg = cfg[vgg_name]
        self.num_classes = num_classes
        self.device = device

        self.features, self.features_masks = self._make_layers(cfg[vgg_name])

        if ('like' not in vgg_name):
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, self.num_classes),
            )

            self.classifier_masks = [torch.ones(512, 512),
                                     torch.ones(512),
                                     torch.ones(512, 512),
                                     torch.ones(512),
                                     torch.ones(self.num_classes, 512),
                                     torch.ones(self.num_classes)]
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, self.num_classes),
            )

            self.classifier_masks = [torch.ones(512, 512),
                                     torch.ones(512),
                                     torch.ones(self.num_classes, 512),
                                     torch.ones(self.num_classes)]                           

        self.apply(_weights_init)

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), out.size(1) * out.size(2) * out.size(3))
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        masks = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                masks += [torch.ones(x, in_channels, 3, 3),
                          torch.ones(x)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
        return nn.Sequential(*layers), masks

    def _weights_to_mask(self):
        features_masks = []
        classifier_masks = []

        step = 4
        if (list(self.named_children())[0][0] == 'features'):
            l = 0
            for name, param in list(self.features.named_parameters())[::step]:
                self.features_masks[l] = 1 * (param.data != 0).float().cpu()
                l += 2

            l = 1
            for name, param in list(self.features.named_parameters())[1::step]:
                self.features_masks[l] = 1 * (param.data != 0).float().cpu()
                l += 2

        l = 0
        for name, param in self.classifier.named_parameters():
            self.classifier_masks[l] = 1 * (param.data != 0).float().cpu()
            l += 1

    def _apply_mask(self):
        step = 4
        if (list(self.named_children())[0][0] == 'features'):
            l = 0
            for name, param in list(self.features.named_parameters())[::step]:
                param.data = param.data * (self.features_masks[l]).to(self.device)
                l += 2

            l = 1
            for name, param in list(self.features.named_parameters())[1::step]:
                param.data = param.data * (self.features_masks[l]).to(self.device)
                l += 2

        l = 0
        for name, param in self.classifier.named_parameters():
            param.data = param.data * (self.classifier_masks[l]).to(self.device)
            l += 1


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, device, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        self.block_masks = self._make_masks(in_planes, planes, stride)

        self.device = device
        self.planes = planes
        self.in_planes = in_planes
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.planes // 4, self.planes // 4),
                                                  "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def _make_masks(self, in_planes, planes, stride):
        return [torch.ones(planes, in_planes, 3, 3), torch.ones(planes, planes, 3, 3), torch.ones(planes)]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)

        out += shortcut[:] * (self.block_masks[-1].reshape((-1, 1, 1)).expand(
            (shortcut.size(-3), shortcut.size(-2), shortcut.size(-1))).to(self.device))
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, device):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.num_blocks = num_blocks
        self.device = device
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_masks = torch.ones(16, 3, 3, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1, self.layer1_masks = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2, self.layer2_masks = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3, self.layer3_masks = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layers_masks = [self.layer1_masks, self.layer2_masks, self.layer3_masks]

        self.linear = nn.Linear(64, num_classes)
        self.linear_masks = [torch.ones(self.num_classes, 64), torch.ones(self.num_classes)]

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers_masks = []
        for stride in strides:
            basicblock = block(self.in_planes, planes, self.device, stride)
            layers.append(basicblock)
            layers_masks.append(basicblock.block_masks)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers), layers_masks

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _save_masks(self, file_name='net_masks.pt'):
        masks_database = {}

        masks_database['conv1.mask'] = self.conv1_masks

        for layer in range(len(self.num_blocks)):
            for block in range(self.num_blocks[layer]):
                for conv_num in range(2):
                    name = 'layer{}.{}.conv{}.mask'.format(layer + 1, block, conv_num + 1)
                    masks_database[name] = self.layers_masks[layer][block][conv_num]

                name = 'layer{}.{}.shortcut.mask'.format(layer + 1, block)
                masks_database[name] = self.layers_masks[layer][block][-1]

        masks_database['linear.weight.mask'] = self.linear_masks[0]
        masks_database['linear.bias.mask'] = self.linear_masks[1]

        torch.save(masks_database, file_name)

    def _load_masks(self, file_name='net_masks.pt'):
        masks_database = torch.load(file_name)

        self.conv1_masks = masks_database['conv1.mask']

        for layer in range(len(self.num_blocks)):
            for block in range(self.num_blocks[layer]):
                for conv_num in range(2):
                    name = 'layer{}.{}.conv{}.mask'.format(layer + 1, block, conv_num + 1)
                    self.layers_masks[layer][block][conv_num] = masks_database[name]

                name = 'layer{}.{}.shortcut.mask'.format(layer + 1, block)
                self.layers_masks[layer][block][-1] = masks_database[name]

        self.linear_masks[0] = masks_database['linear.weight.mask']
        self.linear_masks[1] = masks_database['linear.bias.mask']

    def _apply_mask(self):
        for name, param in list(self.named_parameters()):
            if (name == 'conv1'):
                param.data = param.data * (self.conv1_masks).to(self.device)
            elif ('linear' in name):
                if ('weight' in name):
                    param.data = param.data * (self.linear_masks[0]).to(self.device)
                else:
                    param.data = param.data * (self.linear_masks[1]).to(self.device)
            else:
                for layer in range(len(self.num_blocks)):
                    for block in range(self.num_blocks[layer]):
                        if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                            param.data = param.data * (self.layers_masks[layer][block][0]).to(self.device)
                        elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                            param.data = param.data * (self.layers_masks[layer][block][1]).to(self.device)


def resnet20(num_classes, device):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, device)


def resnet32(num_classes, device):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, device)


def resnet44(num_classes, device):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, device)


def resnet56(num_classes, device):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, device)


def resnet110(num_classes, device):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, device)