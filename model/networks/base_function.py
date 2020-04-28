import re
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from .external_function import SpectralNorm
import torch.nn.functional as F
# from model.networks.markovattn2d_package.markovattn2d import MarkovAttn2d
from model.networks.resample2d_package.resample2d import Resample2d
# from model.networks.correlation_package.correlation   import Correlation
from model.networks.block_extractor.block_extractor  import BlockExtractor
from model.networks.local_attn_reshape.local_attn_reshape   import LocalAttnReshape
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
######################################################################################
# base function for network structure
######################################################################################


# def init_weights(net, init_type='normal', gain=0.02):
#     """Get different initial method for the network weights"""
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, 0.02)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)
class ResBlock3DEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlock3DEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv3d(input_nc,  hidden_nc, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)), use_spect)
        conv2 = spectral_norm(nn.Conv3d(hidden_nc, output_nc, kernel_size=(3,4,4), stride=(1,2,2), padding=(0,1,1)), use_spect)
        bypass = spectral_norm(nn.Conv3d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(nn.AvgPool3d(kernel_size=(3,2,2), stride=(1,2,2)),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out  

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic feature map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segfeature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segfeature = F.interpolate(segfeature, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segfeature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
        
    def hook(self, x, segfeature):
        normalized = self.param_free_norm(x)
        actv = self.mlp_shared(segfeature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out, gamma, beta




class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta

        return out


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'adain':
        norm_layer = functools.partial(ADAIN)
    elif norm_type == 'spade':
        norm_layer = functools.partial(SPADE, config_text='spadeinstance3x3')        
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class EncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(EncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        conv1 = coord_conv(input_nc,  output_nc, use_spect, use_coord, **kwargs_down)
        conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, **kwargs_fine)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1, 
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlock, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc
        self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
        conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)

        if self.learnable_shortcut:
            bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
            self.shortcut = nn.Sequential(bypass,)


    def forward(self, x):
        if self.learnable_shortcut:
            out = self.model(x) + self.shortcut(x)
        else:
            out = self.model(x) + x
        return out

class ResBlocks(nn.Module):
    """docstring for ResBlocks"""
    def __init__(self, num_blocks, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, 
                 nonlinearity= nn.LeakyReLU(), learnable_shortcut=False, use_spect=False, use_coord=False):
        super(ResBlocks, self).__init__()
        hidden_nc = input_nc if hidden_nc is None else hidden_nc
        output_nc = input_nc if output_nc is None else output_nc

        self.model=[]
        if num_blocks == 1:
            self.model += [ResBlock(input_nc, output_nc, hidden_nc, 
                           norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        else:
            self.model += [ResBlock(input_nc, hidden_nc, hidden_nc, 
                           norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            for i in range(num_blocks-2):
                self.model += [ResBlock(hidden_nc, hidden_nc, hidden_nc, 
                               norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]
            self.model += [ResBlock(hidden_nc, output_nc, hidden_nc, 
                            norm_layer, nonlinearity, learnable_shortcut, use_spect, use_coord)]

        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


class SelfAttentionBlock(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttentionBlock,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class GlobalAttentionBlock(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(GlobalAttentionBlock,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, source, target):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = source.size()
        proj_query  = self.query_conv(source).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(target).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(source).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + target
        return out,attention
        

class BilinearSamplingBlock(nn.Module):
    """docstring for BilinearSamplingBlock"""
    def __init__(self):
        super(BilinearSamplingBlock, self).__init__()

    def forward(self, source, flow_field):
        [b,_,h,w] = source.size()
        # flow_field = torch.nn.functional.interpolate(flow_field, (w,h))
        x = torch.arange(w).view(1, -1).expand(h, -1)
        y = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x,y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid/(w-1) - 1
        flow_field = 2*flow_field/(w-1)
        grid = (grid+flow_field).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source, grid)    
        return warp    
        
class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class ResBlockEncoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out        

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    # def __init__(self, fin, fout, opt):
    #     super().__init__()

    def __init__(self, input_nc, output_nc, hidden_nc, label_nc, spade_config_str='spadeinstance3x3', nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False, learned_shortcut=False):
        super(SPADEResnetBlock, self).__init__()        
        # Attributes
        self.learned_shortcut = (input_nc != output_nc) or learned_shortcut
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1), use_spect)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False), use_spect)

        # define normalization layers
        self.norm_0 = SPADE(spade_config_str, input_nc, label_nc)
        self.norm_1 = SPADE(spade_config_str, hidden_nc, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, input_nc, label_nc)


    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s


class ADAINResnetBlock(nn.Module):
    # def __init__(self, fin, fout, opt):
    #     super().__init__()

    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False, learned_shortcut=False):
        super(ADAINResnetBlock, self).__init__()        
        # Attributes
        self.learned_shortcut = (input_nc != output_nc) or learned_shortcut
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=3, stride=1, padding=1), use_spect)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False), use_spect)

        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        if self.learned_shortcut:
            self.norm_s = ADAIN(input_nc, feature_nc)


    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, z))
        else:
            x_s = x
        return x_s

class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out
        
class Jump(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1)

    def forward(self, x):
        out = self.model(x)
        return out

class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W*H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention

class MarkovAttnBlock(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, feature_nc, markov_kernel_size=3):
        super(MarkovAttnBlock, self).__init__()

        # nhidden = 128
        # ks = 5
        # pw = 2
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(structure_nc, nhidden, kernel_size=ks, padding=pw),
        #     nn.ReLU(),
        #     nn.Conv2d(nhidden, feature_nc, kernel_size=ks, padding=pw),
        #     nn.ReLU()
        # )


        self.feature = nn.Conv2d(feature_nc, feature_nc//2, kernel_size=1)
        self.structure = nn.Conv2d(feature_nc, feature_nc//2, kernel_size=1)
        corr_size = 20
        self.corr = Correlation(pad_size=corr_size, kernel_size=1, max_displacement=20, 
                                stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        corr_in_nc = (corr_size+1)**2
        hidden_nc = 128
        self.pre_process = nn.Sequential(
                            nn.Conv2d(corr_in_nc,hidden_nc,kernel_size=3,stride=1,padding=1),
                            nn.LeakyReLU(0.1,inplace=True))

        self.predict_flow = nn.Conv2d(hidden_nc,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.deform_conv  = nn.Conv2d(hidden_nc,2*markov_kernel_size*markov_kernel_size,kernel_size=3,stride=1,padding=1,bias=True)
        self.deform_mask  = nn.Sequential(
                            nn.Conv2d(hidden_nc,markov_kernel_size*markov_kernel_size,kernel_size=3,stride=1,padding=1,bias=True),
                            nn.LeakyReLU(0.1, inplace=True))

        self.markov_attn = MarkovAttn2d(kernel_size=markov_kernel_size)


    def forward(self, source, target):

        f_target = self.structure(target)
        f_source = self.feature(source)
        out_corr = self.corr(f_target, f_source)
        out_corr = self.corr_activation(out_corr)
        hidden = self.pre_process(out_corr)
        flow_fields = self.predict_flow(hidden)
        offset = self.deform_conv(hidden)
        mask = self.deform_mask(hidden)
        output = self.markov_attn(source, flow_fields, offset, mask)

        return output, flow_fields, offset, mask

class ExtractorAttn(nn.Module):
    def __init__(self, feature_nc, kernel_size=4, nonlinearity=nn.LeakyReLU(), softmax=None):
        super(ExtractorAttn, self).__init__()
        self.kernel_size=kernel_size
        hidden_nc = 128
        softmax = nonlinearity if softmax is None else nn.Softmax(dim=1)

        self.extractor = BlockExtractor(kernel_size=kernel_size)
        self.reshape = LocalAttnReshape()
        self.fully_connect_layer = nn.Sequential(
                nn.Conv2d(2*feature_nc, hidden_nc, kernel_size=kernel_size, stride=kernel_size, padding=0),
                nonlinearity,
                nn.Conv2d(hidden_nc, kernel_size*kernel_size, kernel_size=1, stride=1, padding=0),
                softmax,)        
    def forward(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return result

    def hook_attn_param(self, source, target, flow_field):
        block_source = self.extractor(source, flow_field)
        block_target = self.extractor(target, torch.zeros_like(flow_field))
        attn_param_ = self.fully_connect_layer(torch.cat((block_target, block_source), 1))
        attn_param = self.reshape(attn_param_, self.kernel_size)
        result = torch.nn.functional.avg_pool2d(attn_param*block_source, self.kernel_size, self.kernel_size)
        return attn_param_, result


class GaussianBlock(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, feature_nc, markov_kernel_size=4):
        super(GaussianBlock, self).__init__()

        # nhidden = 128
        # ks = 5
        # pw = 2
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(structure_nc, nhidden, kernel_size=ks, padding=pw),
        #     nn.ReLU(),
        #     nn.Conv2d(nhidden, feature_nc, kernel_size=ks, padding=pw),
        #     nn.ReLU()
        # )


        self.feature = nn.Conv2d(feature_nc, feature_nc//2, kernel_size=1)
        self.structure = nn.Conv2d(feature_nc, feature_nc//2, kernel_size=1)
        corr_size = 20
        self.corr = Correlation(pad_size=corr_size, kernel_size=1, max_displacement=20, 
                                stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        corr_in_nc = (corr_size+1)**2
        hidden_nc = 128
        self.pre_process = nn.Sequential(
                            nn.Conv2d(corr_in_nc,hidden_nc,kernel_size=3,stride=1,padding=1),
                            nn.LeakyReLU(0.1,inplace=True))

        self.predict_flow = nn.Conv2d(hidden_nc,2,kernel_size=3,stride=1,padding=1,bias=True)
        # self.deform_conv  = nn.Conv2d(hidden_nc,2*markov_kernel_size*markov_kernel_size,kernel_size=3,stride=1,padding=1,bias=True)
        # self.deform_mask  = nn.Sequential(
        #                     nn.Conv2d(hidden_nc,markov_kernel_size*markov_kernel_size,kernel_size=3,stride=1,padding=1,bias=True),
        #                     nn.LeakyReLU(0.1, inplace=True))

        self.resample2d = Resample2d(kernel_size=markov_kernel_size, sigma=2)


    def forward(self, source, target):

        f_target = self.structure(target)
        f_source = self.feature(source)
        out_corr = self.corr(f_target, f_source)
        out_corr = self.corr_activation(out_corr)
        hidden = self.pre_process(out_corr)
        flow_fields = self.predict_flow(hidden)
        # offset = self.deform_conv(hidden)
        # mask = self.deform_mask(hidden)
        output = self.resample2d(source, flow_fields)

        return output, flow_fields

class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d,  nonlinearity= nn.LeakyReLU(),
                 use_spect=False):
        super(LinearBlock, self).__init__()
        use_bias = True

        self.fc = spectral_norm(nn.Linear(input_nc, output_nc, bias=use_bias), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.fc)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.fc)


    def forward(self, x):
        out = self.model(x)
        return out   
             
class LayerNorm1d(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm1d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
          return F.layer_norm(x, normalized_shape)  


class ADALN1d(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()
        nhidden = 128
        use_bias=True
        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):
        normalized_shape = x.size()[1:]

        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        gamma = gamma.view(*gamma.size()[:2], 1)
        beta = beta.view(*beta.size()[:2], 1)
        out = F.layer_norm(x, normalized_shape) * (1 + gamma)+beta  

        return out 
