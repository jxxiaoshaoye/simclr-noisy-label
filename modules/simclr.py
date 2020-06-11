import torch.nn as nn
import torchvision
import torch
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
import math


def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.fc=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.fc(h)
        return logit




class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x_i) = ResNet(x_i) where hi Rd is the output after the average pooling layer.
    """

    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args

        self.encoder = self.get_net(args.net)

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)(W(1)h_i) where  is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, args.projection_dim, bias=False),
        )
        

    def get_net(self, name):
        nets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
            "9cnn":CNN(),
        }
        if name not in nets.keys():
            raise KeyError(f"{name} is not a valid Net version")
        return nets[name]


    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z


class simclrnet(nn.Module):
    def __init__(self, args, model,n_classes,in_feature):
        super(simclrnet, self).__init__()

        self.args = args
        self.encoder = model
        
        self.n_features = in_feature # get dimensions of fc layer
        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_features, n_classes, bias=True),
        )


    def forward(self, x):
        h = self.encoder(x)
        output = self.classifier(h)

        #if self.args.normalize:
            #z = nn.functional.normalize(z, dim=1)
        return output