import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from model import common
import torch.nn.init as init


def make_model(args, parent=False):
    return BARN(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class BCA(nn.Module):
    def __init__(self, in_ch=256, reduction=16):
        super(BCA, self).__init__()
        self.in_ch = in_ch

        self.reduction = nn.Sequential(nn.Conv2d(self.in_ch, reduction, kernel_size=1, padding=0, stride=1, bias=True),
                                       nn.ReLU(inplace=True))
        self.rowwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=reduction, out_channels=reduction * 4, kernel_size=(1, reduction), stride=1,
                      padding=0, groups=reduction),
            nn.LeakyReLU(0.1))

        self.expand = nn.Sequential(nn.Conv2d(reduction * 4, self.in_ch, kernel_size=1, padding=0, stride=1, bias=True),
                                    nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()
        x = self.reduction(x)  # Bx32xHxW

        x1 = x.view(batch_size, x.size()[1], -1)
        x2 = x1.clone()
        x2 = x2.permute(0, 2, 1)

        corr = torch.matmul(x1, x2).unsqueeze(1)  # Bx1x32x32
        corr = corr.permute(0, 3, 1, 2)  # Bx32x1x32

        res = self.rowwise_conv(corr)  # Bx(32x4)x1x1
        res = self.expand(res)  # Bx256x1x1

        return res * x_org


class BSA(nn.Module):
    def __init__(self, in_ch=256, bsize=8, reduction=16):
        super(BSA, self).__init__()
        self.in_ch = in_ch
        self.bsize = bsize

        self.reduction = nn.Sequential(nn.Conv2d(self.in_ch, reduction, kernel_size=1, padding=0, stride=1, bias=True),
                                       nn.ReLU(inplace=True))

        self.rowwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=bsize ** 2, out_channels=bsize ** 2 * 4, kernel_size=(1, bsize ** 2), stride=1,
                      padding=0, groups=bsize ** 2),
            nn.LeakyReLU(0.1))

        self.expand = nn.Sequential(
            nn.Conv2d(bsize ** 2 * 4, self.bsize ** 2, kernel_size=1, padding=0, stride=1, bias=True, groups=1),
            nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()
        x = self.reduction(x)  # BxRxHxW
        x = F.adaptive_avg_pool2d(x, (self.bsize, self.bsize))  # BxRx16x16

        x1 = x.view(batch_size, x.size()[1], -1)
        x2 = x1.clone()
        x1 = x1.permute(0, 2, 1)

        corr = torch.matmul(x1, x2).unsqueeze(1)
        corr = corr.permute(0, 3, 1, 2)

        res = self.rowwise_conv(corr)
        res = self.expand(res)
        res = res.view(batch_size, 1, x.size(2), x.size(3))
        res = nn.functional.interpolate(res, size=x_org.size()[2:], mode='nearest')

        return res * x_org


class AttBlock(nn.Module):
    def __init__(self, n_feats, bsize, reduction, aggregate='cat', bn=False):
        super(AttBlock, self).__init__()
        self.aggregate = aggregate

        self.channelwise = BCA(n_feats, reduction)
        self.spatialwise = BSA(n_feats, bsize, reduction)

        if self.aggregate == 'cat':
            modules = [nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, bias=True)]
            if bn:
                modules += [nn.BatchNorm2d(n_feats)]
            else:
                modules += [nn.LeakyReLU(0.1)]

            self.conv = nn.Sequential(*modules)
            if isinstance(self.conv._modules['1'], nn.BatchNorm2d):
                init.constant_(self.conv._modules['1'].weight, 0.)
                init.constant_(self.conv._modules['1'].bias, 1.)

    def forward(self, x):
        x_org = x.clone()
        res1 = self.channelwise(x)
        res2 = self.spatialwise(x)

        if self.aggregate == 'cat':
            x = torch.cat((res1, res2), dim=1)
            res = self.conv(x)
            return res + x_org
        else:
            return res1 + res2


class BARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(BARN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        n_seconds = args.n_seconds
        bsize = args.bsize
        reduction = args.reduction
        aggregate = args.aggregate
        self.shift_mean = args  .shift_mean
        scale = args.scale[0]
        act = nn.ReLU(True)
        m = common._Covariance(256, 64)

        if self.shift_mean:
            self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range)
            self.add_mean = common.MeanShift(args.n_colors, args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size), m]

        # define body module
        m_body = []
        for i in range(1, n_resblocks + 1):
            m_body += [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
            if i % (n_resblocks // n_seconds) == 0:
                m_body += [AttBlock(n_feats, bsize, reduction, aggregate, args.att_batchnorm)]

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.shift_mean:
            x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if self.shift_mean:
            x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# class BARN(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(BARN, self).__init__()
#
#         n_resgroups = 4
#         n_resblocks = 4
#         n_feats = args.n_feats
#         kernel_size = 3
#         scale = args.scale[0]
#         act = nn.ReLU(True)
#
#         # RGB mean for DIV2K
#         self.sub_mean = common.MeanShift(args.rgb_range)
#
#         # define head module
#         modules_head = [conv(args.n_colors, n_feats, kernel_size), act]
#
#         # define body module
#         ext_list = [None, None, None, None]
#         # ext_list = [None, None, None, None]
#         modules_body = []
#         for i in range(n_resgroups):
#             modules_body += [ResidualGroup(conv, n_feats, act=act, n_resblocks=n_resblocks, extent=ext_list[i])]
#
#         modules_body.append(conv(n_feats, n_feats, kernel_size))
#
#         # define tail module
#         modules_tail = [
#
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, args.n_colors, kernel_size)]
#
#         self.add_mean = common.MeanShift(args.rgb_range, sign=1)
#
#         self.head = nn.Sequential(*modules_head)
#         self.body = nn.Sequential(*modules_body)
#         self.tail = nn.Sequential(*modules_tail)
#
#     def forward(self, x):
#         x = self.sub_mean(x)
#         x = self.head(x)
#
#         res = self.body(x)
#         res += x
#
#         x = self.tail(res)
#         x = self.add_mean(x)
#
#         return x
#
#     def load_state_dict(self, state_dict, strict=True):
#         own_state = self.state_dict()
#         for name, param in state_dict.items():
#             if name in own_state:
#                 if isinstance(param, nn.Parameter):
#                     param = param.data
#                 try:
#                     own_state[name].copy_(param)
#                 except Exception:
#                     if name.find('tail') == -1:
#                         raise RuntimeError('While copying the parameter named {}, '
#                                            'whose dimensions in the model are {} and '
#                                            'whose dimensions in the checkpoint are {}.'
#                                            .format(name, own_state[name].size(), param.size()))
#             elif strict:
#                 if name.find('tail') == -1:
#                     raise KeyError('unexpected key "{}" in state_dict'
#                                    .format(name))
