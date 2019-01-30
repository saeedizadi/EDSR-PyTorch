import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from model import common
import torch.nn.init as init


def make_model(args, parent=False):
    return SORAN(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class BCA(nn.Module):
    def __init__(self, in_ch=256, reduction=64):
        super(BCA, self).__init__()
        self.in_ch = in_ch

        self.reduction = nn.Sequential(nn.Conv2d(self.in_ch, reduction, kernel_size=1, padding=0, stride=1, bias=True, groups=1),
                                       nn.ReLU(inplace=True))
        self.rowwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=reduction, out_channels=reduction * 4, kernel_size=(1, reduction), stride=1,
                      padding=0, groups=reduction),
            nn.LeakyReLU(0.1))

        self.expand = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()
        x = self.reduction(x)  # Bx128xHxW

        x1 = x.view(batch_size, x.size()[1], -1) # Bx128xHW
        x2 = x1.clone() # Bx128xHW
        x2 = x2.permute(0, 2, 1) # BxHWx128

        corr = torch.matmul(x1, x2).unsqueeze(1) # Bx1x128x128
        corr = corr.permute(0, 3, 1, 2) # Bx128x1x128

        res = self.rowwise_conv(corr) # Bx512x1x1
        res = self.expand(res) # Bx256x1x1

        return res * x_org


class BSA(nn.Module):
    def __init__(self, in_ch=256, bsize=16, reduction=64):
        super(BSA, self).__init__()
        self.in_ch = in_ch
        self.bsize = bsize

        self.reduction = nn.Sequential(nn.Conv2d(self.in_ch, reduction, kernel_size=1, padding=0, stride=1, bias=True),
                                       nn.ReLU(inplace=True))

        self.rowwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=bsize**2, out_channels=reduction * 4, kernel_size=(1, bsize**2), stride=1,
                      padding=0, groups=bsize**2),
            nn.LeakyReLU(0.1))

        # self.expand = nn.Sequential(nn.Conv2d(1, self.in_ch, kernel_size=1, padding=0, stride=1, bias=True),
        #                             nn.Sigmoid())
        self.expand = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()
        # x = self.reduction(x)  # Bx16xHxW
        x = F.adaptive_avg_pool2d(x, (self.bsize, self.bsize))  # Bx64x16x16
        x1 = x.view(batch_size, x.size()[1], -1) # Bx64x256
        x2 = x1.clone()  # Bx64x256
        x1 = x1.permute(0, 2, 1)  # Bx256x64

        corr = torch.matmul(x1, x2).unsqueeze(1)  # Bx1x256x256
        corr = corr.permute(0, 3, 1, 2)  # Bx256x1x256

        res = self.rowwise_conv(corr) # Bx256x1x1
        res = res.view(batch_size, 1, x.size(2), x.size(3)) # Bx1x16x16
        res = nn.functional.interpolate(res, size=x_org.size()[2:], mode='nearest') # Bx1x16x16

        res = self.expand(res)

        return res * x_org


class AttBlock(nn.Module):
    def __init__(self, n_feats):
        super(AttBlock, self).__init__()
        self.channelwise = BCA(n_feats)
        self.spatialwise = BSA(n_feats)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, bias=True),
                                  nn.BatchNorm2d(n_feats))
                                  # nn.LeakyReLU(0.1))

        init.constant_(self.conv._modules['1'].weight, 0.)
        init.constant_(self.conv._modules['1'].bias, 1.)

    def forward(self, x):        
        res1 = self.channelwise(x)
        res2 = self.spatialwise(x)

        x = torch.cat((res1, res2), dim=1)
        res = self.conv(x)

        return res


class ResGroup(nn.Module):
    def __init__(self, conv, n_resblocks, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResGroup, self).__init__()

        modules_body = [
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale) for _ in range(n_resblocks)]

        modules_body.append(AttBlock(n_feats))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class SORAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SORAN, self).__init__()

        n_resblocks = args.n_resblocks
        n_resgroups = args.n_resgroups
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range)
        self.add_mean = common.MeanShift(args.n_colors, args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        for i in range(n_resgroups):
            m_body += [ResGroup(conv, n_resblocks, n_feats, kernel_size, act=act, res_scale=args.res_scale)]

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
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
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