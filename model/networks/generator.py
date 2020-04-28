import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.resample2d_package.resample2d import Resample2d
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

######################################################################################################
# Human Pose Image Generation 
######################################################################################################
class PoseGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(PoseGenerator, self).__init__()
        self.source = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)
        self.target = PoseTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, 
                                                norm, activation, attn_layer, extractor_kz, use_spect, use_coord)
        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                    attn_layer=attn_layer, norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)       

    def forward(self, source, source_B, target_B):
        feature_list = self.source(source)
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        image_gen = self.target(target_B, feature_list, flow_fields, masks)

        return image_gen, flow_fields, masks  

    def forward_hook_function(self, source, source_B, target_B):
        feature_list = self.source(source)
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        hook_target, hook_source, hook_attn, hook_mask = self.target.forward_hook_function(target_B, feature_list, flow_fields, masks)        
        return hook_target, hook_source, hook_attn, hook_mask



class PoseSourceNet(BaseNetwork):
    def __init__(self, input_nc=3, ngf=64, img_f=1024, layers=6, norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):  
        super(PoseSourceNet, self).__init__()
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # encoder part CONV_BLOCKS
        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)        


    def forward(self, source):
        feature_list=[source]
        out = self.block0(source)
        feature_list.append(out)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list


class PoseTargetNet(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(PoseTargetNet, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        self.block0 = EncoderBlock(structure_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         


        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                         nonlinearity, use_spect, use_coord))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer, 
                                             nonlinearity, False, use_spect, use_coord),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                             nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)

            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def forward(self, target_B, source_feature, flow_fields, masks):
        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                out_attn = model(source_feature[i], out, flow_fields[counter])        
                out = out*(1-masks[counter]) + out_attn*masks[counter]
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image

    def forward_hook_function(self, target_B, source_feature, flow_fields, masks):
        hook_target=[]
        hook_source=[]      
        hook_attn=[]      
        hook_mask=[]      
        out = self.block0(target_B)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                attn_param, out_attn = model.hook_attn_param(source_feature[i], out, flow_fields[counter])        
                out = out*(1-masks[counter]) + out_attn*masks[counter]

                hook_target.append(out)
                hook_source.append(source_feature[i])
                hook_attn.append(attn_param)
                hook_mask.append(masks[counter])
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return hook_target, hook_source, hook_attn, hook_mask    


class PoseFlowNet(nn.Module):
    """docstring for FlowNet"""
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(PoseFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 2*structure_nc + image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult,  norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up = ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, 
                                    nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)


    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        inputs = torch.cat((source, source_B, target_B), 1) 
        out = self.block0(inputs)
        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_fields.append(flow_field)
                masks.append(mask)

        return flow_fields, masks

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)
        return flow, mask  

class PoseFlowNetGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,  img_f=1024, layers=6, norm='batch',
                activation='ReLU', encoder_layer=5, attn_layer=[1,2], use_spect=True, use_coord=False):  
        super(PoseFlowNetGenerator, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        self.flow_net = PoseFlowNet(image_nc, structure_nc, ngf, img_f, 
                        encoder_layer, attn_layer=attn_layer,
                        norm=norm, activation=activation, 
                        use_spect=use_spect, use_coord= use_coord)

    def forward(self, source, source_B, target_B):
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        return flow_fields, masks
        
######################################################################################################
# Pose-Guided Person Image Animation
######################################################################################################        
class DanceGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(DanceGenerator, self).__init__()
        self.source_previous = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)
        self.source_reference = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)        
        self.target = FaceTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, 
                                    norm, activation, attn_layer, extractor_kz, use_spect, use_coord)

        flow_norm, flow_activation = 'instance', 'LeakyReLU'
        self.flow_net_previous = PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                            attn_layer=attn_layer, norm=flow_norm, activation=flow_activation,
                                            use_spect=use_spect, use_coord=use_coord)  

        self.flow_net_reference= PoseFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                            attn_layer=attn_layer, norm=flow_norm, activation=flow_activation,
                                            use_spect=use_spect, use_coord=use_coord)                                                     

    def forward(self, BP_frame_step, P_reference, BP_reference, P_previous, BP_previous):
        n_frames_load = BP_frame_step.size(1)
        out_image_gen,out_flow_fields,out_masks,P_previous_recoder=[],[],[],[]

        for i in range(n_frames_load):
            BP = BP_frame_step[:,i,...]
            P_previous  = P_reference  if P_previous  is None else  P_previous
            BP_previous = BP_reference if BP_previous is None else  BP_previous
            P_previous_recoder.append(P_previous)

            previous_feature_list = self.source_previous(P_previous)
            reference_feature_list = self.source_reference(P_reference)

            # Sources_image = torch.cat((P_previous, P_reference ), 0)
            # Sources_bone  = torch.cat((BP_previous,BP_reference), 0)
            # Targets_bone  = torch.cat((BP, BP), 0)

            flow_fields_p, masks_p = self.flow_net_previous( P_previous,  BP_previous,  BP)
            flow_fields_r, masks_r = self.flow_net_reference(P_reference, BP_reference, BP)
            flow,mask=[],[]
            for i in range(len(flow_fields_p)):
                flow.append(flow_fields_p[i])
                flow.append(flow_fields_r[i])
                mask.append(masks_p[i]) 
                mask.append(masks_r[i]) 
            image_gen = self.target(BP, previous_feature_list, reference_feature_list, flow, mask)
            P_previous = image_gen
            BP_previous = BP

            out_image_gen.append(image_gen)
            out_flow_fields.append(flow)
            out_masks.append(mask)
        return out_image_gen, out_flow_fields, out_masks, P_previous_recoder  



class KPInput2DGenerator(BaseNetwork):
    def __init__(self, structure_nc=18, channels=256, layers=4):  
        super(KPInput2DGenerator, self).__init__()
        self.kp_input = KPInputNet2D(keypoint_nc=structure_nc, channels=channels, layers=layers)

       
    def forward(self, input_2d):
        keypoints = self.kp_input(input_2d)
        return keypoints


class KPInputNet2D(BaseNetwork):
    def __init__(self, keypoint_nc=25, channels=256, layers=3, dropout=0.15, image_size=(256,256)):
        super(KPInputNet2D, self).__init__()
        kernel_size=3
        self.keypoint_nc = keypoint_nc
        self.image_size = image_size
        self.layers = layers
        self.expand_conv = nn.Conv1d(keypoint_nc*2, channels, kernel_size, bias=False)
        self.expand_ln = LayerNorm1d(channels)

        self.shrink = nn.Conv1d(channels, keypoint_nc*2, 1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)  
        self.lrelu = nn.LeakyReLU(0.1)  

        layers_conv = []
        layers_ln = []
        self.pad=[(kernel_size - 1) // 2]
        next_dilation = kernel_size
        for i in range(1, self.layers):
            self.pad.append((kernel_size - 1)*next_dilation // 2)
            layers_conv.append(nn.Conv1d(channels,channels,kernel_size,
                                         dilation=next_dilation,bias=False))
            layers_ln.append(ADALN1d(channels, channels))
            layers_conv.append(nn.Conv1d(channels, channels, 1,  dilation=1, bias=False))
            layers_ln.append(ADALN1d(channels, channels))

            next_dilation *= kernel_size
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_ln = nn.ModuleList(layers_ln)

        self.feature_conv_1 = nn.Conv1d(keypoint_nc*2, channels, kernel_size, stride=2 ,bias=False)
        self.feature_conv_2 = nn.Conv1d(channels, channels, kernel_size, stride=2 ,bias=False)
        self.feature_conv_3 = nn.Conv1d(channels, channels, kernel_size, stride=2 ,bias=False)

    def forward(self, kp):
        feature = self.lrelu(self.feature_conv_1(kp))
        feature = self.lrelu(self.feature_conv_2(feature))
        feature = self.lrelu(self.feature_conv_3(feature))
        feature = torch.mean(feature, 2)


        x = self.drop(self.lrelu(self.expand_ln(self.expand_conv(kp))))
        for i in range(self.layers - 1):
            pad = self.pad[i+1]
            res = x[:, :, pad : x.shape[2] - pad]
            x = self.drop(self.lrelu(self.layers_ln[2*i](self.layers_conv[2*i](x), feature)))
            x = res + self.drop(self.lrelu(self.layers_ln[2*i+1](self.layers_conv[2*i+1](x), feature)))
        
        x = self.shrink(x)
        return x


######################################################################################################
# Face Image Generation 
######################################################################################################        
class FaceGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(FaceGenerator, self).__init__()
        self.source_previous = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)
        self.source_reference = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)        
        self.target = FaceTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, 
                                                norm, activation, attn_layer, extractor_kz, use_spect, use_coord)
        self.flow_net = FaceFlowNet(image_nc, structure_nc, ngf=32, img_f=256, encoder_layer=5, 
                                    attn_layer=attn_layer, norm=norm, activation=activation,
                                    use_spect=use_spect, use_coord=use_coord)       

    def forward(self, BP_frame_step, P_reference, BP_reference, P_previous, BP_previous):
        n_frames_load = BP_frame_step.size(1)
        out_image_gen,out_flow_fields,out_masks,P_previous_recoder=[],[],[],[]

        for i in range(n_frames_load):
            # BP_previous = BP_frame_step[:, i, ...]
            BP = BP_frame_step[:,i,...]
            P_previous  = P_reference  if P_previous  is None else  P_previous
            BP_previous = BP_reference if BP_previous is None else  BP_previous
            P_reference = P_reference
            BP_reference = BP_reference
            P_previous_recoder.append(P_previous)

            previous_feature_list = self.source_previous(P_previous)
            reference_feature_list = self.source_reference(P_reference)

            flow_fields, masks = self.flow_net(BP, P_previous, BP_previous, P_reference, BP_reference)
            image_gen = self.target(BP, previous_feature_list, reference_feature_list, flow_fields, masks)
            P_previous = image_gen
            BP_previous = BP

            out_image_gen.append(image_gen)
            out_flow_fields.append(flow_fields)
            out_masks.append(masks)
        return out_image_gen, out_flow_fields, out_masks, P_previous_recoder         


class FaceTargetNet(BaseNetwork):

    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(FaceTargetNet, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        self.block0 = EncoderBlock(structure_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         


        # decoder part
        mult = min(2 ** (layers-1), img_f//ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                         nonlinearity, use_spect, use_coord))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer, 
                                             nonlinearity, False, use_spect, use_coord),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                             nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)

            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn_p' + str(i), attn)

                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn_r' + str(i), attn)


                # attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                # setattr(self, 'attn' + str(i), attn)                

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def forward(self, BP, previous_feature_list, reference_feature_list, flow_fields, masks):
        out = self.block0(BP)
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) 

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model_p = getattr(self, 'attn_p' + str(i))
                model_r = getattr(self, 'attn_r' + str(i))

                out_attn_p = model_p(previous_feature_list[i], out, flow_fields[2*counter])        
                out_attn_r = model_r(reference_feature_list[i], out, flow_fields[2*counter+1])        
                out_p = out*(1-masks[2*counter])   + out_attn_p*masks[2*counter]
                out_r = out*(1-masks[2*counter+1]) + out_attn_r*masks[2*counter+1]
                out = out_p + out_r 
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image


class FaceFlowNet(nn.Module):
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(FaceFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        input_nc = 3*structure_nc + 2*image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult,  norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)         
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up = ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, 
                                    nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:
                flow_out = nn.Conv2d(ngf*mult, 4, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)


    def forward(self, BP, P_previous, BP_previous, P_reference, BP_reference):
        flow_fields=[]
        masks=[]
        inputs = torch.cat((BP, P_previous, BP_previous, P_reference, BP_reference), 1) 
        out = self.block0(inputs)
        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_field_p, flow_field_r = torch.split(flow_field, 2, dim=1)
                mask_p, mask_r = torch.split(mask, 1, dim=1)
                flow_fields.append(flow_field_p)
                flow_fields.append(flow_field_r)
                masks.append(mask_p)
                masks.append(mask_r)

        return flow_fields, masks

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)

        model = getattr(self, 'mask' + str(i))
        mask = model(out)

        return flow, mask     

######################################################################################################
# Shape Net Image Generation (Multi-view synthesis)
######################################################################################################        
class ShapeNetGenerator(BaseNetwork):
    def __init__(self,  image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU',   attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):
        super(ShapeNetGenerator, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        self.source = PoseSourceNet(image_nc, ngf, img_f, layers, 
                                                    norm, activation, use_spect, use_coord)
        self.target = ShapeNetTargetNet(image_nc, structure_nc, output_nc, ngf, img_f, layers, num_blocks, 
                                                norm, activation, attn_layer, extractor_kz, use_spect, use_coord)
        self.flow_net = ShapeNetFlowNet(image_nc, structure_nc, 32, 256, 
                                        encoder_layer=5, attn_layer=attn_layer,
                                        norm=norm, activation=activation, 
                                        use_spect=use_spect, use_coord= use_coord)

    def forward(self, source, source_B, target_B):
        feature_list = self.source(source)
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        image_gen = self.target(target_B, feature_list, flow_fields, masks)        
        return image_gen, flow_fields, masks    


class ShapeNetTargetNet(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, img_f=1024, layers=6, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[1,2], extractor_kz={'1':5,'2':5}, use_spect=True, use_coord=False):  
        super(ShapeNetTargetNet, self).__init__()

        self.layers = layers
        self.attn_layer = attn_layer

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)


        self.block0 = ResBlockDecoder(structure_nc, ngf, None, norm_layer, 
                         nonlinearity, use_spect, use_coord)
        mult = min(2 ** (layers-1), img_f//ngf)
        self.block1 = ResBlockDecoder(ngf, ngf*mult, None, norm_layer, 
                             nonlinearity, use_spect, use_coord)
        
        for i in range(layers):
            mult_prev = mult
            mult = min(2 ** (layers-i-2), img_f//ngf) if i != layers-1 else 1
            if num_blocks == 1:
                up = nn.Sequential(ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                         nonlinearity, use_spect, use_coord))
            else:
                up = nn.Sequential(ResBlocks(num_blocks-1, ngf*mult_prev, None, None, norm_layer, 
                                             nonlinearity, False, use_spect, use_coord),
                                   ResBlockDecoder(ngf*mult_prev, ngf*mult, None, norm_layer, 
                                             nonlinearity, use_spect, use_coord))
            setattr(self, 'decoder' + str(i), up)

            if layers-i in attn_layer:
                attn = ExtractorAttn(ngf*mult_prev, extractor_kz[str(layers-i)], nonlinearity, softmax=True)
                setattr(self, 'attn' + str(i), attn)

        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spect, use_coord)


    def forward(self, target_B, source_feature, flow_fields, masks):
        target_B = target_B.repeat(1, 1, 8, 8)
        out = self.block0(target_B)
        out = self.block1(out)

        counter=0
        for i in range(self.layers):
            if self.layers-i in self.attn_layer:
                model = getattr(self, 'attn' + str(i))

                out_attn = model(source_feature[i], out, flow_fields[counter])        
                out = out*(1-masks[counter]) + out_attn*masks[counter]
                counter += 1

            model = getattr(self, 'decoder' + str(i))
            out = model(out)

        out_image = self.outconv(out)
        return out_image     

class ShapeNetFlowNet(nn.Module):
    def __init__(self, image_nc, structure_nc, ngf=64, img_f=1024, encoder_layer=5, attn_layer=[1], norm='batch',
                activation='ReLU', use_spect=True, use_coord=False):
        super(ShapeNetFlowNet, self).__init__()

        self.encoder_layer = encoder_layer
        self.decoder_layer = encoder_layer - min(attn_layer)
        self.attn_layer = attn_layer
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # input_nc = structure_nc + image_nc
        input_nc = image_nc

        self.block0 = EncoderBlock(input_nc, ngf, norm_layer,
                                 nonlinearity, use_spect, use_coord)
        mult = 1
        for i in range(encoder_layer-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ngf)
            block = EncoderBlock(ngf*mult_prev, ngf*mult,  norm_layer,
                                 nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)   

        self.cat = ResBlocks(1, ngf*mult+structure_nc, ngf*mult, None, norm_layer, nonlinearity, False, use_spect, use_coord)
        
        for i in range(self.decoder_layer):
            mult_prev = mult
            mult = min(2 ** (encoder_layer-i-2), img_f//ngf) if i != encoder_layer-1 else 1
            up = ResBlockDecoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, 
                                    nonlinearity, use_spect, use_coord)
            setattr(self, 'decoder' + str(i), up)
            
            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spect, use_coord)
            setattr(self, 'jump' + str(i), jumpconv)

            if encoder_layer-i-1 in attn_layer:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'output' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)


    def forward(self, source, source_B, target_B):
        flow_fields=[]
        masks=[]
        # inputs = torch.cat((source), 1) 
        out = self.block0(source)
        result=[out]
        for i in range(self.encoder_layer-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 

        out = self.encode_ShapeNet_bone(source_B, target_B, out)    
        for i in range(self.decoder_layer):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layer-i-2])
            out = out+jump

            if self.encoder_layer-i-1 in self.attn_layer:
                flow_field, mask = self.attn_output(out, i)
                flow_fields.append(flow_field)
                masks.append(mask)

        return flow_fields, masks

    def attn_output(self, out, i):
        model = getattr(self, 'output' + str(i))
        flow = model(out)
        model = getattr(self, 'mask' + str(i))
        mask = model(out)

        return flow, mask   

    def encode_ShapeNet_bone(self, source_B, target_B, out):
        B=source_B-target_B
        _,_,w,h = out.size()
        B=B.repeat(1, 1, w, h)
        out = torch.cat((out,B), 1) 
        out = self.cat(out)  
        return out   

class ShapeNetFlowNetGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64,  img_f=1024, norm='batch',
                activation='ReLU', encoder_layer=5, attn_layer=[1,2], use_spect=True, use_coord=False):  
        super(ShapeNetFlowNetGenerator, self).__init__()

        self.attn_layer = attn_layer

        self.flow_net = ShapeNetFlowNet(image_nc, structure_nc, ngf, img_f, 
                        encoder_layer, attn_layer=attn_layer,
                        norm=norm, activation=activation, 
                        use_spect=use_spect, use_coord= use_coord)

    def forward(self, source, source_B, target_B):
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        return flow_fields, masks 


