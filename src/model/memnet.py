"""MemNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import PIL


def make_model(args, parent=False):
    return MEMNET(args)

class MEMNET(nn.Module):
    def __init__(self, args):
        super(MEMNET, self).__init__()
        n_colors = args.n_colors
        n_feats = args.n_feats
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks

        self.feature_extractor = BNReLUConv(n_colors, n_feats)
        self.reconstructor = BNReLUConv(n_feats, n_colors)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(n_feats, n_resblocks, i + 1) for i in range(n_resgroups)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual

        return out


class MemoryBlock(nn.Module):
    """Note: n_resgroups denotes the number of MemoryBlock currently"""

    def __init__(self, n_feats, n_resblocks, n_resgroups):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(n_feats) for i in range(n_resblocks)]
        )
        self.gate_unit = BNReLUConv((n_resblocks + n_resgroups) * n_feats, n_feats, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, n_feats, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(n_feats, n_feats, k, s, p)
        self.relu_conv2 = BNReLUConv(n_feats, n_feats, k, s, p)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, n_colors, n_feats, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(n_colors))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(n_colors, n_feats, k, s, p, bias=False))