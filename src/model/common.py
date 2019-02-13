import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class SecondOrderDownsample(nn.Module):
    def __init__(self, in_features, reduction, down_factor):
        super(SecondOrderDownsample, self).__init__()
        self.reduction = reduction
        self.down_factor = down_factor
        self.in_features = in_features
        # self.norm = nn.InstanceNorm1d(in_features, affine=False)
        if self.reduction <= self.in_features:
            self.reduce = nn.Conv2d(in_features, reduction, kernel_size=1, padding=0, stride=1)
            self.rowwise_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, reduction), stride=1,
                                          padding=0)
            self.expansion = nn.Conv2d(reduction, in_features, kernel_size=1, padding=0, stride=1)
        else:
            self.rowwise_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, in_features), stride=1,
                                          padding=0)

    def chunk(self, x, downsample):
        batchsize = x.size(0)
        channels = x.size(1)
        res = [torch.split(c, downsample, dim=3) for c in list(torch.split(x, downsample, dim=2))]
        covs = []
        for i in range(len(res)):
            for j in range(len(res[0])):
                block = res[i][j].contiguous().view(batchsize, channels, -1)  # NxCxHW
                # block = self.norm(block)
                # onev = torch.ones((batchsize, N, 1)).to(torch.device('cuda'))
                # I = torch.eye(N).to(torch.device('cuda')) - (1 / N) * torch.matmul(
                #     onev, torch.transpose(onev, 1, 2))
                # temp = torch.matmul(block, I)
                # covs.append(torch.matmul(temp, block.permute(0, 2, 1)).unsqueeze(1))  # Nx1xCxC
                cc = torch.matmul(block, block.permute(0, 2, 1)).unsqueeze(1)
                covs.append(cc)  # Nx1xCxC
        covs = torch.cat(covs, dim=1)  # NxM^2xCxC
        return covs

    def forward(self, x):
        batchsize = x.size(0)
        M = 1 if self.down_factor == 1 else x.size(3) // self.down_factor
        if self.reduction < self.in_features:
            x = self.reduce(x)
        x = self.chunk(x, x.size(2)) if self.down_factor == 1 else self.chunk(x, self.down_factor)  # NxM^2xCxC
        if batchsize == 1:
            print(x.size())
        MM = x.size(1)  # M^2

        x = x.view(batchsize * MM, 1, x.size(2), x.size(3))  # (NxM^2)x1xCxC
        x = self.rowwise_conv(x)
        x = x.view(batchsize, MM, x.size(2))  # NxM^2xC

        x = x.permute(0, 2, 1)  # NxCxM^2
        x = x.view(batchsize, self.reduction, M, M)  # NxCxMxM
        if self.reduction < self.in_features:
            x = self.expansion(x)
        return x


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
