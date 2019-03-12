## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def make_model(args, parent=False):
    return CBAM(args)


class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()

        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = torch.mean(x, 1, keepdim=True, out=None)
        x_max, _ = torch.max(x, 1, keepdim=True, out=None)
        y = torch.cat((x_avg, x_max), dim=1)
        y = self.conv_du(y)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv2d(channel, reduction, 1, padding=0, bias=True),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv2d(reduction, channel, 1, padding=0, bias=True)

    def forward(self, x):
        y_avg = self.fc2(self.fc1(self.avg(x)))
        y_max = self.fc2(self.fc1(self.max(x)))

        y = F.sigmoid(y_avg + y_max)

        return x * y


class CBAMBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(CBAMBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.ca = CALayer(n_feat)
        self.sa = SALayer(n_feat)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        c_res = self.ca(res)
        res = self.sa(c_res)
        res += x
        return res


class CBAM(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(CBAM, self).__init__()

        res_scale = args.res_scale
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        # self.sub_mean = common.MeanShift(args.n_colors, args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            CBAMBlock(conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=res_scale) \
            for _ in range(n_resblocks)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean = common.MeanShift(args.n_colors, args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        print("Num of parameters: {0}".format(self.get_numparams()))

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def get_numparams(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
