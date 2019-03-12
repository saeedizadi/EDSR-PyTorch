import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class SecondOrderDownsample(nn.Module):
    def __init__(self, in_features):
        super(SecondOrderDownsample, self).__init__()

        self.rowwise_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=(1, in_features),
                                      stride=1,
                                      padding=0, groups=in_features)
        self.norm = nn.InstanceNorm1d(in_features, affine=False)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = x.view(batch_size, x.size()[1], -1)  # NxCxHW
        x1 = self.norm(x1)
        x2 = x1.permute(0, 2, 1)  # NxHWxC

        corr = torch.matmul(x1, x2).unsqueeze(1)  # Bx1x64x64
        corr = corr.permute(0, 3, 1, 2)  # Bx64x1x64

        x = self.rowwise_conv(corr)  # Bx64x1x1
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, n_colors, rgb_range, sign=-1):
        if n_colors == 3:
            mean = (0.4488, 0.4371, 0.4040)
            super(MeanShift, self).__init__(3, 3, kernel_size=1)
            self.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.bias.data = sign * rgb_range * torch.Tensor(mean)
        else:
            mean = [0.44]
            super(MeanShift, self).__init__(1, 1, kernel_size=1)
            self.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.bias.data = sign * rgb_range * torch.tensor(mean)

        for p in self.parameters():
            p.requires_grad = False


# class MeanShift(nn.Conv2d):
#     def __init__(
#             self, rgb_range,
#             rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
#         super(MeanShift, self).__init__(3, 3, kernel_size=1)
#         std = torch.Tensor(rgb_std)
#         self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
#         self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
#         for p in self.parameters():
#             p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
