"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from model.networks.base_network import BaseNetwork
# from models.networks.loss import *
from model.networks.discriminator import *
from model.networks.generator import *
from model.networks.encoder import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'model.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


# def modify_commandline_options(parser, is_train):
#     opt, _ = parser.parse_known_args()

#     netG_cls = find_network_using_name(opt.netG, 'generator')
#     parser = netG_cls.modify_commandline_options(parser, is_train)
#     if is_train:
#         netD_cls = find_network_using_name(opt.netD, 'discriminator')
#         parser = netD_cls.modify_commandline_options(parser, is_train)
#     netE_cls = find_network_using_name('conv', 'encoder')
#     parser = netE_cls.modify_commandline_options(parser, is_train)

#     return parser


def create_network(cls, opt, **parameter_dic):
    net = cls(**parameter_dic)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type)
    return net



def define_g(opt, name=None, **parameter_dic):
    name = opt.netG if name is None else name
    netG_cls = find_network_using_name(name, 'generator')
    return create_network(netG_cls, opt, **parameter_dic)


def define_d(opt, name=None, **parameter_dic):
    name = opt.netD if name is None else name
    netD_cls = find_network_using_name(name, 'discriminator')
    return create_network(netD_cls, opt, **parameter_dic)


def define_e(opt, name=None, **parameter_dic):
    # there exists only one encoder type
    name = opt.netE if name is None else name
    netE_cls = find_network_using_name(name, 'encoder')
    return create_network(netE_cls, opt, **parameter_dic)
