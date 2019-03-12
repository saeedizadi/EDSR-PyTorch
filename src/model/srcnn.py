import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn
import torchvision.transforms as transforms
import PIL


def make_model(args, parent=False):
    return SRCNN(args)


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='relu',
                 norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class SRCNN(torch.nn.Module):
    def __init__(self, args):
        super(SRCNN, self).__init__()

        n_colors = args.n_colors
        n_feats = args.n_feats
        self.scale = args.scale[0]
        self.layers = torch.nn.Sequential(
            ConvBlock(n_colors, n_feats, 9, 1, 4, norm=None),
            ConvBlock(n_feats, n_feats // 2, 5, 1, 2, norm=None),
            ConvBlock(n_feats // 2, n_colors, 5, 1, 2, activation=None, norm=None)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
