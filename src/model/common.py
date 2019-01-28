import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class NonLocal(nn.Module):
    def __init__(self, in_channels, extent=None, downsample='conv'):
        super(NonLocal, self).__init__()
        self.extent = extent

        if self.extent is not None:
            if downsample == 'conv':
                gather_modules = [
                    nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2,
                                            padding=1),
                                  nn.ReLU(inplace=True))
                    for _ in range(int(math.log(extent, 2)))]
            elif downsample == 'avg':
                self.gather = nn.AvgPool2d(kernel_size=extent, stride=extent)
            else:
                raise NotImplementedError

            self.gather = nn.Sequential(*gather_modules)
            # self.gather = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
            #                         dilation=extent, stride=extent)

        self.in_channels = in_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)

        if self.extent is not None:
            x_org = x.clone()
            x = self.gather(x)

        g_x = self.g(x).view(batch_size, self.in_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.in_channels, *x.size()[2:])
        W_y = self.W(y)

        if self.extent is not None:
            W_y = nn.functional.interpolate(W_y, size=x_org.size()[2:], mode='nearest')
            x = x_org

        # z = W_y + x
        z = torch.sigmoid(W_y) * x

        return z


class MeanShift(nn.Conv2d):
    def __init__(self, mean, sign=-1):
        if len(mean) == 3:
            super(MeanShift, self).__init__(3, 3, kernel_size=1)
            self.weight.data = torch.eye(3).view(3, 3, 1, 1)
            self.bias.data = sign * torch.Tensor(mean)
        else:
            super(MeanShift, self).__init__(1, 1, kernel_size=1)
            self.weight.data = torch.eye(1).view(1, 1, 1, 1)
            self.bias.data = sign * torch.tensor(mean)

        for p in self.parameters():
            p.requires_grad = False


# class MeanShift(nn.Conv2d):
#     def __init__(
#         self, rgb_range,
#         rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
#
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
