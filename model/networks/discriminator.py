import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
import util.util as util
from model.networks.base_function import *
# from model.networks.external_function import SpectralNorm
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

class ResDiscriminator(BaseNetwork):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoder(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out


class PatchDiscriminator(BaseNetwork):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """
    def __init__(self, input_nc=3, ndf=64, img_f=512, layers=3, norm='batch', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2 ** i, img_f // ndf)
            sequence +=[
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]

        mult_prev = mult
        mult = min(2 ** i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out            

class TemporalDiscriminator(BaseNetwork):
    """
    Temporal Discriminator Network
    input_length: number of input image frames

    """
    def __init__(self, input_nc=3, input_length=6, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(TemporalDiscriminator, self).__init__()

        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # self.pool = nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # encoder part
        self.block0 = ResBlock3DEncoder(input_nc, 1*ndf, 1*ndf, norm_layer, nonlinearity, use_spect, use_coord)
        self.block1 = ResBlock3DEncoder(1*ndf,    2*ndf, 1*ndf, norm_layer, nonlinearity, use_spect, use_coord)

        feature_len = input_length-4
        mult = 2*feature_len
        for i in range(layers - 2):
            mult_prev = mult
            mult = min(2 ** (i + 2), img_f//ndf)
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))

    def forward(self, x):
        # [b,l,c,h,w] = x.shape
        out = self.block0(x)
        out = self.block1(out)
        [b,c,l,h,w] = out.shape
        out = out.view(b,-1,h,w)
        for i in range(self.layers - 2):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out


