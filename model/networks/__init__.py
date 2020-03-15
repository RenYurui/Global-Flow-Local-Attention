import torch
from model.networks.base_network import BaseNetwork
from model.networks.discriminator import *
from model.networks.generator import *
from model.networks.encoder import *
import util.util as util


def find_network_using_name(target_network_name, filename, extend=None):
    extend=filename if extend is None else extend
    target_class_name = target_network_name + extend
    module_name = 'model.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network



def create_network(cls, opt, **parameter_dic):
    net = cls(**parameter_dic)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type)
    return net



def define_g(opt, name=None, filename=None, **parameter_dic):
    name = opt.netG if name is None else name
    filename = 'generator' if filename is None else filename
    netG_cls = find_network_using_name(name, filename, extend='generator')
    return create_network(netG_cls, opt, **parameter_dic)


def define_d(opt, name=None, filename=None, **parameter_dic):
    name = opt.netD if name is None else name
    filename = 'discriminator' if filename is None else filename
    netD_cls = find_network_using_name(name, filename)
    return create_network(netD_cls, opt, **parameter_dic)


def define_e(opt, name=None, **parameter_dic):
    # there exists only one encoder type
    name = opt.netE if name is None else name
    netE_cls = find_network_using_name(name, 'encoder')
    return create_network(netE_cls, opt, **parameter_dic)
