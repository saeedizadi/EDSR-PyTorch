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
    def __init__(self, in_ch=64, reduction=16):
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
        x = self.reduction(x)  # Bx64xHxW

        x1 = x.view(batch_size, x.size()[1], -1)
        x2 = x1.clone()
        x2 = x2.permute(0, 2, 1)

        corr = torch.matmul(x1, x2).unsqueeze(1)
        corr = corr.permute(0, 3, 1, 2)

        res = self.rowwise_conv(corr)
        res = self.expand(res)

        return res * x_org


class BSA(nn.Module):
    def __init__(self, in_ch=64, reduction=16):
        super(BSA, self).__init__()
        self.in_ch = in_ch

        self.reduction = nn.Sequential(nn.Conv2d(self.in_ch, reduction, kernel_size=1, padding=0, stride=1, bias=True),
                                       nn.ReLU(inplace=True))

        self.downsample = nn.AvgPool2d(kernel_size=6)
        self.rowwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=reduction * 4, kernel_size=(1, 64), stride=1,
                      padding=0, groups=reduction),
            nn.LeakyReLU(0.1))

        self.expand = nn.Sequential(nn.Conv2d(1, self.in_ch, kernel_size=1, padding=0, stride=1, bias=True),
                                    nn.Sigmoid())

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()
        x = self.reduction(x)  # Bx16xHxW
        x = self.downsample(x)

        x1 = x.view(batch_size, x.size()[1], -1)
        x2 = x1.clone()
        x1 = x1.permute(0, 2, 1)

        corr = torch.matmul(x1, x2).unsqueeze(1)
        corr = corr.permute(0, 3, 1, 2)

        res = self.rowwise_conv(corr)
        res = res.view(batch_size, 1, x.size(2), x.size(3))
        res = nn.functional.interpolate(res, size=x_org.size()[2:], mode='nearest')

        res = self.expand(res)

        return res * x_org


class AttBlock(nn.Module):
    def __init__(self, n_feats):
        super(AttBlock, self).__init__()
        self.channelwise = BCA(n_feats)
        self.spatialwise = BSA(n_feats)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=1, bias=True),
                                  nn.BatchNorm2d(n_feats),
                                  nn.ReLU(inplace=True))

        init.constant_(self.conv._modules['1'].weight, 0.)
        init.constant_(self.conv._modules['1'].bias, 1.)

    def forward(self, x):
        x_org = x.clone()
        res1 = self.channelwise(x)
        res2 = self.spatialwise(x)

        x = torch.cat((res1, res2), dim=1)
        res = self.conv(x)

        return res + x_org


class NonLocal(nn.Module):
    def __init__(self, in_channels, extent=1, downsample='conv'):
        super(NonLocal, self).__init__()
        self.extent = extent

        if self.extent != 1:
            if downsample == 'conv':
                gather_modules = [
                    nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=2,
                                            stride=2,
                                            groups=in_channels))
                    for _ in range(int(math.log(extent, 2)))]

                self.gather = nn.Sequential(*gather_modules)
            elif downsample == 'avg':
                self.gather = nn.AvgPool2d(kernel_size=extent, stride=extent)

            elif downsample == 'max':
                self.gather = nn.MaxPool2d(kernel_size=extent, stride=extent)

            else:
                raise NotImplementedError

        self.in_channels = in_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                         kernel_size=1, stride=1, padding=0))
        self.batch_norm = nn.BatchNorm2d(self.in_channels)
        init.constant_(self.batch_norm.weight, 0)
        init.constant_(self.batch_norm.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        x_org = x.clone()

        if self.extent != 1:
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

        if self.extent != 1:
            y = nn.functional.interpolate(y, size=x_org.size()[2:], mode='nearest')

        W_y = self.batch_norm(self.W(y))
        return W_y + x_org


#


class BARN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(BARN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        mean = [0.44]
        self.sub_mean = common.MeanShift(mean)
        self.add_mean = common.MeanShift(mean, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        ext_list = args.ex_ratios
        m_body = []
        for i in range(n_resblocks):
            m_body += [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
            if i % 4 == 0:
                m_body += [AttBlock(n_feats)]
                # ex = ext_list.pop(0)
                # if ex is not None:
                #     m_body += [NonLocal(n_feats, ex)]

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
