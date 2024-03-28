########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import math


class NConvUNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_channels=2, pos_fn='SoftPlus'):
        super(NConvUNet, self).__init__()


        self.nconv1 = NConv2d(in_ch, in_ch * num_channels, (5, 5), pos_fn, 'k',
                              padding=(2, 2))  # richard: weird error, used to be 2
        self.nconv2 = NConv2d(in_ch * num_channels, in_ch * num_channels, (5, 5), pos_fn, 'k', padding=(2, 2))
        self.nconv3 = NConv2d(in_ch * num_channels, in_ch * num_channels, (5, 5), pos_fn, 'k', padding=(2, 2))

        self.nconv4 = NConv2d(2 * in_ch * num_channels, in_ch * num_channels, (3, 3), pos_fn, 'k', padding=(1, 1))
        self.nconv5 = NConv2d(2 * in_ch * num_channels, in_ch * num_channels, (3, 3), pos_fn, 'k', padding=(1, 1))
        self.nconv6 = NConv2d(2 * in_ch * num_channels, in_ch * num_channels, (3, 3), pos_fn, 'k', padding=(1, 1))

        self.nconv7 = NConv2d(in_ch * num_channels, out_ch, (1, 1), pos_fn, 'k')

    def forward(self, x0, c0):
        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1)

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).to(x0.get_device())
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c1_ds /= 4

        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)

        # Downsample 2
        ds = 2
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)

        x2_dss = torch.zeros(c2_dss.size()).to(x0.get_device())
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i, j, :, :] = x2_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_dss /= 4

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)

        # Downsample 3
        ds = 2
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)

        x3_dss = torch.zeros(c3_dss.size()).to(x0.get_device())
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i, j, :, :] = x3_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_dss /= 4
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)

        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds, x4), 1), torch.cat((c3_ds, c4), 1))

        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds, x34), 1), torch.cat((c2_ds, c34), 1))

        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')
        xout, cout = self.nconv6(torch.cat((x23, x1), 1), torch.cat((c23, c1), 1))

        xout, cout = self.nconv7(xout, cout)

        return xout, cout


# Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='n', stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):

        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, (0, 0),
                                      groups, bias, padding_mode='zeros')

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / s
        cout = cout.view(sz)

        return nconv, cout

    def enforce_pos(self):
        p = self.weight
        if self.pos_fn.lower() == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0], p_sz[1], -1)
            p = F.softmax(p, -1).data
            self.weight.data = p.view(p_sz)
        elif self.pos_fn.lower() == 'exp':
            self.weight.data = torch.exp(p).data
        elif self.pos_fn.lower() == 'softplus':
            self.weight.data = F.softplus(p, beta=10).data
        elif self.pos_fn.lower() == 'sigmoid':
            self.weight.data = F.sigmoid(p).data
        else:
            print('Undefined positive function!')
            return

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n':  # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))
        elif self.init_method == 'p':  # Poisson
            mu = self.kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)

        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels) + 0.01)


class EnforcePos(object):
    def __init__(self, name, pos_fn):
        self.name = name
        self.pos_fn = pos_fn

    def compute_weight(self, module):
        return _pos(getattr(module, self.name + '_p'), self.pos_fn)

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(name, pos_fn)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        #
        module.register_parameter(name + '_p', Parameter(_pos(weight, pos_fn).data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_p']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def _pos(p, pos_fn):
    pos_fn = pos_fn.lower()
    if pos_fn == 'softmax':
        p_sz = p.size()
        p = p.view(p_sz[0], p_sz[1], -1)
        p = F.softmax(p, -1)
        return p.view(p_sz)
    elif pos_fn == 'exp':
        return torch.exp(p)
    elif pos_fn == 'softplus':
        return F.softplus(p, beta=10)
    elif pos_fn == 'sigmoid':
        return F.sigmoid(p)
    else:
        print('Undefined positive function!')
        return


def remove_weight_pos(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, EnforcePos) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))


# UNet with SoftPlus activation
class UNetSP(nn.Module):
    def __init__(self, n_channels, n_classes, m=8):
        super(UNetSP, self).__init__()
        self.inc = inconv(n_channels, m * 4)
        self.down1 = down(m * 4, m * 4)
        self.down2 = down(m * 4, m * 8)
        self.down3 = down(m * 8, m * 8)
        # self.down4 = down(128, 128)
        # self.up1 = up(256, 64)
        self.up2 = up(m * 8 + m * 8, m * 8)
        self.up3 = up(m * 8 + m * 4, m * 4)
        self.up4 = up(m * 4 + m * 4, m * 4)
        self.outc = outconv(m * 4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 32
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 64
        x4 = self.down3(x3)  # 128
        # x5 = self.down4(x4) #128
        # x = self.up1(x5, x4) #128
        x = self.up2(x4, x3)  # 128
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.softplus(x)

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class PNCNN(nn.Module):
    def __init__(self, nclasses):
        super(PNCNN, self).__init__()

        self.conf_estimator = UNetSP(5, 5)
        self.nconv = NConvUNet(5, 5)
        self.var_estimator = UNetSP(5, 5)

    def forward(self, x0):
        # x0 = x0['proj'][:,:1,:,:]  # use only depth
        # x0 = x0['proj']
        c0 = self.conf_estimator(x0) # confidence
        xout, cout = self.nconv(x0, c0) # output and output con
        cout = self.var_estimator(cout)
        out = torch.cat((xout, cout), 1) # 4, 10, 64, 2048, output, output confidence, and initial confidence
        return out