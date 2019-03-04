import torch
import os
import glob
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import copy
import pathlib
import argparse
import measures
import collections
import math
from torchvision import transforms, datasets

# define cross entropy loss with optional weight intialization regularization
def cross_entropy_loss(model, init_model, output, target, strength):
    loss = nn.CrossEntropyLoss(output, target)
    for w, w0 in zip(model.parameters(), init_model.parameters()):
        loss += strength * torch.pow(torch.abs(w - w0), 2)
    return loss

# train the model for one epoch on the given set
def train(model, init_model, device, train_loader, optimizer):
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.size(0), -1), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = cross_entropy_loss(model, init_model, output, target, FLAGS.init_reg_strength)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# evaluate the model on the given set
def validate(model, init_model, device, val_loader):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device).view(data.size(0), -1), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * cross_entropy_loss(model, init_model, output, target, FLAGS.init_reg_strength)

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i,:].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
        val_margin = np.percentile( margin.cpu().numpy(), 5 )

    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset), val_margin


# Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir):

    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset

# This function trains a fully connected neural net with a single hidden layer on the given dataset and calculates
# various measures on the learned network
def train_model(args):
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    nchannels, nclasses = 3, 10
    if args["dataset"] == 'MNIST': nchannels = 1
    if args["dataset"] == 'CIFAR100': nclasses = 100

    # create an initial model
    model = nn.Sequential(nn.Linear(32 * 32 * nchannels, args["nunits"]), nn.ReLU(), nn.Linear(args["nunits"], nclasses))
    model = model.to(device)

    # create a copy of the initial model to be used later
    init_model = copy.deepcopy(model)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), args["learningrate"], momentum=args["momentum"], weight_decay=args["weightdecay"])

    # loading data
    train_dataset = load_data('train', args["dataset"], args["datadir"])
    val_dataset = load_data('val', args["dataset"], args["datadir"])

    train_loader = DataLoader(train_dataset, batch_size=args["batchsize"], shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args["batchsize"], shuffle=False, **kwargs)
   
    start_epoch = 0

    path = "saved_models/" + args["dataset"] + "/WD" + str(args["weightdecay"]) + "/N" + str(int(math.log(args["nunits"], 2)))
    if os.path.isdir(path):
        # If exact epochs dir exists, select it
        # Else find latest directory
        epoch_path = path + "/E" + str(args["epochs"])
        if os.path.isdir(epoch_path):
            latest_checkpoint = epoch_path + "/checkpoint.pth.tar"
        else:
            latest_dir = max(glob.glob(os.path.join(path, '*/')), key=os.path.getmtime)
            latest_checkpoint = latest_dir + "/checkpoint.pth.tar"

        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch']
        epoch = start_epoch
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        init_model.load_state_dict(checkpoint['init'])
        print("Loading checkpoint for model: " + str(int(math.log(args['nunits'], 2))) + " epoch " + str(epoch))

    # training the model
    for epoch in range(start_epoch, args["epochs"]):
        # train for one epoch
        tr_err, tr_loss = train(model, init_model, device, train_loader, optimizer)

        val_err, val_loss, val_margin = validate(model, init_model, device, val_loader)

        print('Epoch: ' + str(epoch + 1) + "/" + str(args["epochs"]) + '\t Training loss: ' + str(round(tr_loss,3)) + '\t', 'Training error: ' + str(round(tr_err,3)) + '\t Validation error: ' + str(round(val_err,3)))

        if (epoch + 1) % 50 == 0 and epoch > 0:
            path = "./saved_models/" + args["dataset"] + "/WD" + str(args["weightdecay"]) + "/N" + str(int(math.log(args["nunits"], 2))) + "/E" + str(epoch + 1)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "init": init_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": (epoch + 1)
                }, path + "/checkpoint.pth.tar") 

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_loss < args["stopcond"]:
            break

    tr_err, tr_loss, tr_margin = validate(model, init_model, device, train_loader)
    val_err, val_loss, val_margin = validate(model, init_model, device, val_loader)
    print('\nFinal: Training loss: ' + str(round(tr_loss,3)) + '\t Training margin: ' + str(round(tr_margin,3)) + '\t Training error: ' + str(round(tr_err,3)) + '\t Validation error: ' + str(round(val_err,3)) + '\n')

    measure = measures.calculate(model, init_model, device, train_loader, tr_margin)
    return measure

if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Training a fully connected NN with one hidden layer')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10)')
    parser.add_argument('--nunits', default=1024, type=int,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--batchsize', default=64, type=int,
                        help='input batch size (default: 64)')
    parser.add_argument('--learningrate', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weightdecay', default=0, type=float,
                        help='weight decay (default: 0)')
    parser.add_argument('--init_reg_strength', default=0, type=float,
                        help='initialization regularization strength (default: 0)')
    args, unparsed = parser.parse_known_args()
    train_model(vars(args))

