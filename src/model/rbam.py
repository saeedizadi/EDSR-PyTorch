## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def make_model(args, parent=False):
    return HAM(args)


class SpatialSO(nn.Module):
    def __init__(self, in_ch=64, bsize=16):
        super(SpatialSO, self).__init__()
        self.in_ch = in_ch
        self.bsize = bsize

        self.rowwise_conv = nn.Conv2d(in_channels=self.bsize ** 2, out_channels=self.bsize ** 2,
                                      kernel_size=(1, bsize ** 2), stride=1, padding=0,
                                      groups=self.bsize ** 2)

    def forward(self, x):
        batch_size = x.size(0)

        _, _, h, w = x.size()

        x = F.adaptive_avg_pool2d(x, (self.bsize, self.bsize))  # BxRx16x16

        x2 = x.view(batch_size, x.size(1), -1)  # BxCx256
        n = x.size(1)
        # x2 = F.instance_norm(x2, momentum=1.0)  # BxCx256
        x1 = x2.permute(0, 2, 1)  # Bx256xR

        corr = torch.bmm(x1, x2).unsqueeze(1) / n
        x = corr.permute(0, 3, 1, 2)  # Nx64x1x64

        x = self.rowwise_conv(x)  # Nx64x1x1

        x = x.view(batch_size, 1, self.bsize, self.bsize)

        x = nn.functional.interpolate(x, size=(h, w), mode='nearest')  # Nx1xHxW
        return x


class SpatialFO(nn.Module):
    def __init__(self):
        super(SpatialFO, self).__init__()
        pass

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True, out=None)  # Nx1xHxW
        return x


class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()

        self.so = SpatialSO(in_ch=channel, bsize=16)
        self.fo = SpatialFO()
        self.conv_du = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, stride=1, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.so(x) + self.fo(x)  # Nx1xHxW
        y = self.conv_du(y)
        return x * y


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
        self.sa = SALayer(n_feat)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
        self.conv = nn.Conv2d(2*n_feat, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        c_res = self.ca(res)
        s_res = self.sa(res)
        res = self.conv(torch.cat((c_res, s_res), dim=1))

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

        print("Num of parameters: {0}".format(self.get_numparams()))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def get_numparams(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
