# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math

def make_model(args, parent=False):
    return SESR(args)


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.rblock = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            256, 16, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            16, 256, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def resBlock(self, x):
        out = self.rblock(x)
        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)
        out = out * out1
        out = self.trans(out)
        out = x + out
        return out

    def forward(self, x):
        depth = 4
        list_out = []
        for i in range(depth):
            x = self.resBlock(x)
        output = self.upsample(x)
        list_out.append(output)
        return list_out


class SESR(nn.Module):
    def __init__(self,args):
        super(SESR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)

        self.convt_I3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F3 = self.make_layer(_Conv_Block)
        print("Num of parameters: {0}".format(self.get_numparams()))


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     if isinstance(m, nn.ConvTranspose2d):
        #         c1, c2, h, w = m.weight.data.size()
        #         weight = get_upsample_filter(h)
        #         m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        HR_2x = []
        HR_4x = []
        HR_8x = []
        for i in range(len(convt_F1)):
            convt_R1 = self.convt_R1(convt_F1[i])
            tmp = convt_I1 + convt_R1
        HR_2x.append(tmp)
        # HR_2x = tmp
        convt_F2 = self.convt_F2(convt_F1[-1])
        convt_I2 = self.convt_I2(HR_2x[-1])
        for j in range(len(convt_F2)):
            convt_R2 = self.convt_R2(convt_F2[j])
            tmp = convt_I2 + convt_R2
        HR_4x.append(tmp)
        # convt_F3 = self.convt_F3(convt_F2[-1])
        # convt_I3 = self.convt_I3(HR_4x[-1])
        # for j in range(len(convt_F3)):
        #    convt_R3 = self.convt_R3(convt_F3[j])
        #    tmp = convt_I3 + convt_R3
        # HR_8x.append(tmp)

        return HR_4x[-1]

    def get_numparams(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
