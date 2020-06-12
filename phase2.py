import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from torch.autograd import Variable
from experiment import ex
from model import load_model
from utils import post_config_hook

from modules import LogisticRegression
from modules.simclr import simclrnet
from modules.transformations import TransformsSimCLR

from cifar import CIFAR10, CIFAR100
from coteachingloss import loss_coteaching
import torch.nn.functional as F
import numpy as np
import shutil
import copy


def train(args, loader, simclr_model,criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = simclr_model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(args, loader, simclr_model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    simclr_model.eval()
    for step, (x, y) in enumerate(loader):
        simclr_model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = simclr_model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

    


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:7" )
    
    # with torch.autograd.set_detect_anomaly(True):
    root = "./datasets"

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root,
            split="train",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.STL10(
            root,
            split="test",
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    elif args.dataset == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=224).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    simclr_model, _, _ = load_model(args, train_loader, reload_model=True)
    in_feature=simclr_model.n_features
    n_classes = 10
    simclr_model=simclrnet(args,simclr_model.encoder,n_classes,in_feature).to(args.device)

    simclr_model.eval()

    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epoch):
        loss_epoch, accuracy_epoch = train(
            args, train_loader, simclr_model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(
        args, arr_test_loader, simclr_model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )