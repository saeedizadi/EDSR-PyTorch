import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn
import torchvision.transforms as transforms
import PIL

def make_model(args, parent=False):
    return DRCN(args)

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


class DRCN(torch.nn.Module):
    def __init__(self, args):
        super(DRCN, self).__init__()

        n_colors = args.n_colors
        n_feats = args.n_feats
        num_recursions = 16
        self.num_recursions = num_recursions
        # embedding layer
        self.embedding_layer = nn.Sequential(
            ConvBlock(n_colors, n_feats, 3, 1, 1, norm=None),
            ConvBlock(n_feats, n_feats, 3, 1, 1, norm=None)
        )

        # conv block of inference layer
        self.conv_block = ConvBlock(n_feats, n_feats, 3, 1, 1, norm=None)

        # reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            ConvBlock(n_feats, n_feats, 3, 1, 1, activation=None, norm=None),
            ConvBlock(n_feats, n_colors, 3, 1, 1, activation=None, norm=None)
        )

        # initial w
        self.w_init = torch.ones(self.num_recursions) / self.num_recursions
        self.w = Variable(self.w_init.cuda(), requires_grad=True)

    def forward(self, x):
        # embedding layer
        h0 = self.embedding_layer(x)

        # recursions
        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))

        y_d_ = []
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d+1]))
            out_sum += torch.mul(y_d_[d], self.w[d])
        out_sum = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))

        # skip connection
        final_out = torch.add(out_sum, x)

        return final_out