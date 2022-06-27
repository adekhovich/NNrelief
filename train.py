import torch
from datetime import datetime

from utils import accuracy, validate, init_model
from dataloaders import get_dataset, load_dataset

import argparse


def train_lenet(train_loader, model, criterion, optimizer, device):
   
    model.train()
    running_loss = 0

    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        
        with torch.no_grad():
            if (list(model.named_children())[0][0] == 'feature_extractor'):
                l = 0
                for name, param in model.feature_extractor.named_parameters():
                    param.grad.data = param.grad.data*(model.feature_extractor_masks[l]).to(device)
                    l += 1
            
            l = 0
            for name, param in model.classifier.named_parameters():
                param.grad.data = param.grad.data*(model.classifier_masks[l]).to(device)
                l += 1    
              

        optimizer.step()
        
        with torch.no_grad():
            if (list(model.named_children())[0][0] == 'feature_extractor'):
                l = 0
                for name, param in model.feature_extractor.named_parameters():
                        param.data = param.data*(model.feature_extractor_masks[l]).to(device)
                        l += 1
            
            l = 0
            for name, param in model.classifier.named_parameters():
                param.data = param.data*(model.classifier_masks[l]).to(device)
                l += 1  

    with torch.no_grad():
        model._apply_mask()             
        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def train_vgg(train_loader, model, criterion, optimizer, device):
   
    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        
        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        
        #with torch.no_grad():
        step=4
        if (list(model.named_children())[0][0] == 'features'):
            l = 0
            for name, param in list(model.features.named_parameters())[::step]:
                param.grad.data = param.grad.data*(model.features_masks[l]).to(device)
                l += 2

            l = 1
            for name, param in list(model.features.named_parameters())[1::step]:
                param.grad.data = param.grad.data*(model.features_masks[l]).to(device)
                l += 2    
                 
            
        l = 0
        for name, param in model.classifier.named_parameters():
            param.grad.data = param.grad.data*(model.classifier_masks[l]).to(device)
            l += 1    
        

        optimizer.step()
        
        
    with torch.no_grad():
        model._apply_mask()
                    
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def train_resnet(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()

        with torch.no_grad():
            for name, param in list(model.named_parameters()):
                if (name == 'conv1'):
                    param.grad.data = param.grad.data * (model.conv1_masks).to(device)
                elif ('linear' in name):
                    if ('weight' in name):
                        param.grad.data = param.grad.data * (model.linear_masks[0]).to(device)
                    else:
                        param.grad.data = param.grad.data * (model.linear_masks[1]).to(device)
                else:
                    for layer in range(len(model.num_blocks)):
                        for block in range(model.num_blocks[layer]):
                            if (name == 'layer{}.{}.conv1.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data * (model.layers_masks[layer][block][0]).to(device)
                            elif (name == 'layer{}.{}.conv2.weight'.format(layer + 1, block)):
                                param.grad.data = param.grad.data * (model.layers_masks[layer][block][1]).to(device)

        optimizer.step()



    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def training_loop(model, criterion, optimizer, scheduler,
                  train_loader, valid_loader, epochs, model_name, 
                  device, file_name='model.pth', print_every=1):
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []

    if 'lenet' in model_name:
        train = train_lenet
    elif 'vgg' in model_name:
        train = train_vgg
    else:
        train = train_resnet    

    # Train model
    print('TRAINING...')
    for epoch in range(0, epochs):
        # training

        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            scheduler.step()

        train_acc = accuracy(model, train_loader, device=device)
        valid_acc = accuracy(model, valid_loader, device=device)

        if valid_acc > best_acc:
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    return model, (train_losses, valid_losses)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.dataset_name == 'mnist':
        train_dataset, test_dataset = get_dataset('mnist')
        num_classes = 10
    else:
        train_dataset, test_dataset = get_dataset(args.dataset_name)
        if args.dataset_name == 'cifar10':
            num_classes = 10
        else:
            num_classes = 100

    train_loader, test_loader = load_dataset(train_dataset, test_dataset, BATCH_SIZE=args.batch_size)        

    net = init_model(args.model_name, device, num_classes)
    torch.save(net.state_dict(), args.path_init_params)

    loss = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.decay_epochs_train,
                                                     gamma=args.gamma)

    net, _ = training_loop(model=net,
                           criterion=loss,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           train_loader=train_loader,
                           valid_loader=test_loader,
                           epochs=args.train_epochs,
                           model_name=args.model_name,
                           device=device,
                           file_name=args.path_pretrained_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mnist', help='dataset to use')
    parser.add_argument('--path_data', type=str, default='./data', help='path to save/load dataset')
    parser.add_argument('--download_data', type=bool, default=True, help='download dataset')
    parser.add_argument('--model_name', type=str, default='lenet5', help='network architecture to use')
    parser.add_argument('--path_pretrained_model', type=str, default='./pretrained_model.pth', help='path to save model')
    parser.add_argument('--path_init_params', type=str, default='./init_params.pth', help='path to save initial parameters')
    parser.add_argument('--batch_size', type=int, default=120, help='number of examples per mini-batch')
    parser.add_argument('--train_epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='multistep', help='learning rate decay type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--decay_epochs_train', nargs='+', type=int, default=[30], help='epochs for multistep decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--seed', type=int, default=0, help='seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    train(args)
    
