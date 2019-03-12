## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def make_model(args, parent=False):
    return HAM(args)





class ChannelSO(nn.Module):
    def __init__(self, in_features):
        super(ChannelSO, self).__init__()

        self.rowwise_conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=(1, in_features),
                                      stride=1,
                                      padding=0, groups=in_features)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.permute(0, 2, 3, 1)
        x2 = x.view(batch_size, -1, x.size()[3])  # NxHWxC
        n = x2.size(1)
        # x2 = F.instance_norm(x2, momentum=1.0)  # NxHWxC
        x1 = x2.permute(0, 2, 1)  # NxCxHW

        corr = torch.bmm(x1, x2).unsqueeze(1)/n  # Bx1x64x64
        corr = corr.permute(0, 3, 1, 2)  # Bx64x1x64

        x = self.rowwise_conv(corr)  # Bx64x1x1

        return x


class ChannelFO(nn.Module):
    def __init__(self):
        super(ChannelFO, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avg(x)
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.so = ChannelSO(channel)
        self.fo = ChannelFO()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fo(x) + self.so(x)
        y = self.conv_du(y)
        return x * y


class HOAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(HOAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.ca = CALayer(n_feat)
        # self.sa = SALayer(n_feat)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        # self.conv = nn.Conv2d(2*n_feat, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)

        # res = self.body(res).mul(self.res_scale)
        res += x
        return res


class HAM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAM, self).__init__()

        res_scale = args.res_scale
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            HOAB(conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=res_scale) \
            for _ in range(n_resblocks)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.n_colors, args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x