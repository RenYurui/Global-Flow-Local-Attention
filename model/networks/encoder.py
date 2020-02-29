"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
# from models.networks.normalization import get_nonspade_norm_layer
from model.networks.base_function import *


class DURGBEncoder(BaseNetwork):
    """docstring for DURGBEncoder"""
    def __init__(self, nc_structure, nc_rgb=3, ngf=64, img_f=1024, layers=6, 
                 norm='none', activation='ReLU', use_spect=True, use_coord=False):
        super(DURGBEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = RGBEncoder(nc_structure, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
    def forward(self, img_m, structure):
        fea_structure = self.encoder_str(structure)
        fea_image = self.encoder_rgb(img_m)
        return fea_image, fea_structure





class RGBStructureEncoder(BaseNetwork):
    """docstring for RGBStructureEncoder"""
    def __init__(self, input_nc_structure, input_nc_rgb=3, ngf=64, z_nc=128, img_f=1024, layers=6, 
                 norm='none', activation='ReLU', use_spect=True, use_coord=False):
        super(RGBStructureEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(input_nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = StructureEncoder(input_nc_structure, ngf, z_nc, img_f,  layers, 
                                            norm, activation, use_spect, use_coord)
    def forward(self, img_m, structure=None):
        if structure is not None:
            _, distribution = self.encoder_str(structure)
            feature = self.encoder_rgb(img_m)
            return feature, distribution
        else:
            feature = self.encoder_rgb(img_m)
            distribution = []

            # prior distribution
            b, _, w, h = feature.shape
            p_mu = torch.zeros(b, z_nc, w, h)
            p_std = torch.ones(b, z_nc, w, h)
            distribution.append([p_mu, p_std])
            return feature, distribution

        
class RGBStructureAddNoneLabelEncoder(BaseNetwork):
    """add new label to strucutres to repesesnt nothing input"""
    def __init__(self, input_nc_structure, input_nc_rgb=3, ngf=64, z_nc=128, img_f=1024, layers=6, 
                 norm='none', activation='ReLU', use_spect=False, use_coord=False):
        super(RGBStructureAddNoneLabelEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(input_nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = StructureEncoder(input_nc_structure+input_nc_rgb+1, ngf, z_nc, img_f,  layers, 
                                            norm, activation, use_spect, use_coord)

    def forward(self, img_m, mask, structure=None):
        if structure is not None:
            inputs = torch.cat((img_m, mask, structure), 1)
            feature_label, distribution = self.encoder_str(inputs)
            feature = self.encoder_rgb(img_m)
            return feature, feature_label, distribution
        else:
            #TODO: make a structure label
            pass
    def feature_forward(self, img_m):
        return self.encoder_rgb(img_m)

    def structure_forward(self, img_m, mask, structure):
        inputs = torch.cat((img_m, mask, structure), 1)
        feature_label, distribution = self.encoder_str(inputs)
        return feature_label, distribution


class RGBSpatioEncoder(BaseNetwork):
    """add new label to strucutres to repesesnt nothing input"""
    def __init__(self, input_nc_structure, input_nc_rgb=3, ngf=64, z_nc=128, img_f=1024, layers=5, s_layers_down=5, s_layers_up=2,
                 norm='none', activation='ReLU', use_spect=False, use_coord=False):
        super(RGBSpatioEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(input_nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = SpatioStructureEncoder(input_nc_structure+input_nc_rgb+1, ngf, z_nc, img_f, s_layers_down, 
                                            s_layers_up, norm, activation, use_spect, use_coord)

    def forward(self, img_m, mask, structure=None):
        if structure is not None:
            inputs = torch.cat((img_m, mask, structure), 1)
            distribution = self.encoder_str(inputs)
            feature = self.encoder_rgb(img_m)
            return feature, distribution
        else:
            #TODO: make a structure label
            pass
    def feature_forward(self, img_m):
        return self.encoder_rgb(img_m)

    def structure_forward(self, img_m, mask, structure):
        inputs = torch.cat((img_m, mask, structure), 1)
        distribution = self.encoder_str(inputs)
        return distribution


class RGBSpatioLargeEncoder(BaseNetwork):
    """add new label to strucutres to repesesnt nothing input"""
    def __init__(self, input_nc_structure, input_nc_rgb=3, ngf=64, z_nc=128, img_f=1024, layers=5, L=1,
                 norm='none', activation='ReLU', use_spect=False, use_coord=False):
        super(RGBSpatioLargeEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(input_nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = SpatioStructureLargeEncoder(input_nc_structure+input_nc_rgb+1, ngf, z_nc, img_f, layers, 
                                            L, norm, activation, use_spect, use_coord)

    def forward(self, img_m, mask, structure=None):
        if structure is not None:
            inputs = torch.cat((img_m, mask, structure), 1)
            distribution = self.encoder_str(inputs)
            feature = self.encoder_rgb(img_m)
            return feature, distribution
        else:
            #TODO: make a structure label
            pass
    def feature_forward(self, img_m):
        return self.encoder_rgb(img_m)

    def structure_forward(self, img_m, mask, structure):
        inputs = torch.cat((img_m, mask, structure), 1)
        distribution = self.encoder_str(inputs)
        return distribution    


class AdainEncoder(BaseNetwork):
    """add new label to strucutres to repesesnt nothing input"""
    def __init__(self, input_nc_structure, input_nc_rgb=3, ngf=64, z_nc=128, img_f=1024, input_size=256,
                layers=5, L=1, norm='none', activation='ReLU', 
                use_spect=False, use_coord=False, use_coord_s=True):
        super(AdainEncoder, self).__init__()
        self.encoder_rgb = RGBEncoder(input_nc_rgb, ngf, img_f, layers, 
                                      norm, activation, use_spect, use_coord)
        self.encoder_str = AdainStructureEncoder(input_nc_structure+input_nc_rgb+1, ngf, z_nc, img_f, 
                                                input_size, layers, L, 
                                                norm, activation, use_spect, use_coord_s)


    def forward(self, img_m, mask, structure=None):
        if structure is not None:
            inputs = torch.cat((img_m, mask, structure), 1)
            distribution = self.encoder_str(inputs)
            feature = self.encoder_rgb(img_m)
            return feature, distribution
        else:
            #TODO: make a structure label
            pass
    def feature_forward(self, img_m):
        return self.encoder_rgb(img_m)

    def structure_forward(self, img_m, mask, structure):
        inputs = torch.cat((img_m, mask, structure), 1)
        distribution = self.encoder_str(inputs)
        return distribution              

class RGBEncoder(BaseNetwork):
    """
    RGB image Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(RGBEncoder, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, img_m):
        """
        :param img_m: image with mask regions I_m
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        return feature

class StructureEncoder(BaseNetwork):
    """
    Structure Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc, ngf=64, z_nc=128, img_f=1024,  layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(StructureEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, structure):

        img = structure

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        distribution = self.obtain_distribution(out)
        return feature, distribution

    def obtain_distribution(self, f_in):
        distribution=[]
        o = self.posterior(f_in)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([p_mu, F.softplus(p_std)])
        return distribution






class SpatioStructureEncoder(BaseNetwork):
    """
    Structure Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc, ngf=64, z_nc=4, img_f=1024, layers_down=5, layers_up=2, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(SpatioStructureEncoder, self).__init__()

        self.layers_down = layers_down
        self.layers_up = layers_up
        self.z_nc = z_nc

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers_down-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        mult = min(2 ** (layers_down - 1), img_f // ngf)
        for i in range(layers_up):
            mult_prev = mult
            mult = min(2 ** (layers_down - i - 2), img_f // ngf)
            upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            jumpconv = Jump(ngf * mult, ngf * mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            setattr(self, 'jump' + str(i), jumpconv)

        self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)

    def forward(self, structure):
        img = structure
        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers_down-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        for i in range(self.layers_up):
            model = getattr(self, 'decoder' + str(i))
            jump =  getattr(self, 'jump' + str(i))
            out = model(out)
            out = out + jump(feature[self.layers_down-2-i])
        # infer part
        distribution = self.obtain_distribution(out)
        return distribution

    def obtain_distribution(self, f_in):
        distribution=[]
        o = self.posterior(f_in)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([p_mu, F.softplus(p_std)])
        return distribution


class SpatioStructureLargeEncoder(BaseNetwork):
    """
    Structure Encoder Network
    the output have the same resulotion with the input
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc, ngf=64, z_nc=64, img_f=1024, layers=5, L=1, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False):
        super(SpatioStructureLargeEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        ngf = z_nc if ngf < z_nc else ngf

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (layers - i - 2), img_f // ngf)
            upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)
            jumpconv = Jump(ngf * mult, ngf * mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            setattr(self, 'jump' + str(i), jumpconv)
        self.blockup = ResBlockDecoder(ngf * mult , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_coord)

        for i in range(L):
            posterior = ResBlock(ngf * mult, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'infer_posterior' + str(i), posterior)
        self.posterior = ResBlock(ngf * mult, 2 * z_nc, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord)


    def forward(self, structure):
        img = structure
        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)



class AdainStructureEncoder(BaseNetwork):
    """
    Structure Encoder Network
    the output have the same resulotion with the input
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc, ngf=64, z_nc=64, img_f=1024, input_size=256, layers=5, L=1, norm='none', activation='ReLU',
                 use_spect=True, use_coord=True):
        super(AdainStructureEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L

        # ngf = z_nc if ngf < z_nc else ngf

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        k_s = (input_size // 2**self.layers)
        self.global_pooling = torch.nn.AvgPool2d(k_s)

        
        for i in range(L):
            posterior = LinearBlock(ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spect)
            setattr(self, 'infer_posterior' + str(i), posterior)
        self.posterior = LinearBlock(ngf*mult, 2*z_nc, norm_layer, nonlinearity, use_spect)


    def forward(self, structure):
        img = structure
        # encoder part
        out = self.block0(img)
        # feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            # feature.append(out)
        out = self.global_pooling(out).view(out.size(0), out.size(1))

        # infer part
        for i in range(self.L):
            model = getattr(self, 'infer_posterior' + str(i))
            out = model(out)
        out = self.posterior(out)
        distribution = self.obtain_distribution(out)
        return distribution

    def obtain_distribution(self, f_in):
        distribution=[]
        p_mu, p_std = torch.split(f_in, self.z_nc, dim=1)
        distribution.append([p_mu, F.softplus(p_std)])
        return distribution

