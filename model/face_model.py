import torch
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import cv2



class Face(BaseModel):
    """
    Face Image Animation using edge image
    """
    def name(self):
        return "Face Image Animation"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='face', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--netD_V', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for the affine regularization loss')
        parser.add_argument('--frames_D_V', type=int, default=3, help='number of frames of D_V')
        

        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(display_freq=100)
        parser.set_defaults(eval_iters_freq=1000)
        parser.set_defaults(print_freq=100)
        parser.set_defaults(save_latest_freq=1000)
        parser.set_defaults(save_iters_freq=10000)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen','correctness_p', 'correctness_r','content_gen','style_gen',
                            'regularization_p', 'regularization_r',
                            'ad_gen','dis_img_gen',
                            'ad_gen_v', 'dis_img_gen_v']

        self.visual_names = ['P_reference','BP_reference', 'P_frame_step','BP_frame_step','img_gen', 'flow_fields', 'masks']        
        self.model_names = ['G','D','D_V']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        # define the Animation model
        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)
        if len(opt.gpu_ids) > 1:
            self.net_G = torch.nn.DataParallel(self.net_G, device_ids=self.gpu_ids)

        self.flow2color = util.flow2color()

        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        if len(opt.gpu_ids) > 1:
            self.net_D = torch.nn.DataParallel(self.net_D, device_ids=self.gpu_ids)

        input_nc = (opt.frames_D_V-1) * opt.image_nc
        self.net_D_V = network.define_d(opt, input_nc=input_nc, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        if len(opt.gpu_ids) > 1:
            self.net_D_V = torch.nn.DataParallel(self.net_D_V, device_ids=self.gpu_ids)                

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_V.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)
        else:
            self.results_dir_base = self.opt.results_dir
        self.setup(opt)


    def set_input(self, data):
        # move to GPU and change data types
        opt = self.opt
        _, n_frames_total, self.height, self.width = data['P'].size()  #        
        self.n_frames_total = n_frames_total // opt.image_nc
        n_frames_load = opt.max_frames_per_gpu                        # number of total frames loaded into GPU at a time for each batch
        self.n_frames_load = min(n_frames_load, self.n_frames_total)
        
        if self.isTrain:
            self.P_reference  = data['P'][:,:opt.image_nc, ...].cuda()
            self.BP_reference = data['BP'][:, :opt.structure_nc, ...].cuda()
            self.P_previous = None
            self.BP_previous = None

        self.P_images = data['P']
        self.BP_structures = data['BP']
        self.image_paths = [path[0] for path in data['P_path']]

        if not self.isTrain:
            assert self.opt.batchSize == 1
            if data['frame_idx'] == self.opt.start_frame + self.opt.n_frames_pre_load_test:
                self.P_previous = None
                self.BP_previous = None
                self.P_reference  = data['P'][:,:opt.image_nc, ...].cuda()
                self.BP_reference = data['BP'][:, :opt.structure_nc, ...].cuda()
                print(self.opt.results_dir)
                if data['change_seq']:
                    self.write2video()
            # else:
            #     self.P_previous = self.test_generated
            #     self.BP_previous = self.test_BP_previous
            self.opt.results_dir = os.path.join(self.results_dir_base,
                                                self.image_paths[0].split('/')[-2])
           

    def write2video(self):
        images = sorted(glob.glob(self.opt.results_dir+'/*_vis.png'))
        image_array=[]
        for image in images:
            image_rgb = cv2.imread(image)
            image_array.append(image_rgb) 

        out_name = self.opt.results_dir+'.avi' 
        print('write video %s'%out_name)  
        height, width, layers = image_rgb.shape
        size = (width,height)
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
        for i in range(len(image_array)):
            out.write(image_array[i])
        out.release()




    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if 'frame_step' in name:
                    value = value[0]
                    list_value=[]
                    for i in range(value.size(0)):
                        list_value.append(value[i].unsqueeze(0))
                    value=list_value
                if 'flow_field' in name or 'masks' in name:
                    list_value = [item for sub_list in value for item in sub_list]
                    value = list_value

                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                    # visual_ret[name] = util.tensor2im(value[-1].data)       
                else:
                    visual_ret[name] =self.convert2im(value, name)         
        return visual_ret 




    def test(self, save_features=False, save_all=False, generate_edge=True):
        """Forward function used in test time"""
        # img_gen, flow_fields, masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
        height, width = self.height, self.width
        image_nc, structure_nc = self.opt.image_nc, self.opt.structure_nc
        n_frames_pre_load = self.opt.n_frames_pre_load_test

        self.BP_frame_step = self.BP_structures.view(-1, n_frames_pre_load, structure_nc, height, width).cuda()
        self.test_generated, self.flow_fields, self.masks, _ = self.net_G(self.BP_frame_step, 
                                                                self.P_reference, 
                                                                self.BP_reference,
                                                                self.P_previous,
                                                                self.BP_previous)
        self.P_previous = self.test_generated[-1] 
        self.BP_previous = self.BP_frame_step[:,-1,... ]   
        
        self.test_generated = torch.cat(self.test_generated, 0)      
        self.save_results(self.test_generated, data_name='vis', data_ext='png')

        if generate_edge:
            value = self.BP_structures[:,0,...].unsqueeze(1)
            value = (1-value)*2-1
            self.save_results(value, data_name='edge', data_ext='png')




    def update(self):
        """Run forward processing to get the inputs"""
        image_nc, structure_nc = self.opt.image_nc, self.opt.structure_nc
        n_frames_total, n_frames_load = self.n_frames_total, self.n_frames_load
        height, width = self.height, self.width
        for i in range(0, n_frames_total, n_frames_load):
            self.P_frame_step  = self.P_images[:, i*image_nc:(i+n_frames_load)*image_nc].cuda()
            self.BP_frame_step = self.BP_structures[:, i*structure_nc:(i+n_frames_load)*structure_nc].cuda()
            self.P_frame_step = self.P_frame_step.view(-1, n_frames_load, image_nc, height, width)
            self.BP_frame_step = self.BP_frame_step.view(-1, n_frames_load, structure_nc, height, width)

            self.img_gen, self.flow_fields, self.masks, self.P_previous_recoder = self.net_G(self.BP_frame_step, 
                                                                    self.P_reference, 
                                                                    self.BP_reference,
                                                                    self.P_previous,
                                                                    self.BP_previous)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

            self.P_previous  = self.img_gen[-1].detach()
            self.BP_previous = self.BP_frame_step[:,-1,...].detach()



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # if print_loss:
            # print(D_real_loss)
            # print(D_fake_loss)
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        real = self.P_frame_step[:,i,...]
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, real, fake)

        base_function._unfreeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        # fake = [self.img_gen[i]]
        # real = [self.P_frame_step[:,i,...]]
        fake = []
        real = []        
        for frame in range(self.opt.frames_D_V-1):
            fake.append(self.img_gen[i+frame]-self.img_gen[i+frame+1])
            real.append(self.P_frame_step[:,i+frame,...]
                       -self.P_frame_step[:,i+frame+1,...])
        fake = torch.cat(fake, dim=1)
        real = torch.cat(real, dim=1)
        self.loss_dis_img_gen_v = self.backward_D_basic(self.net_D_V, real, fake)

    def backward_G(self):
        """Calculate training loss for the generator"""
        # gen_tensor = torch.cat([v.unsqueeze(1) for v in self.img_gen], 1)
        loss_style_gen, loss_content_gen, loss_app_gen=0,0,0

        for i in range(len(self.img_gen)):
            gen = self.img_gen[i]
            gt = self.P_frame_step[:,i,...]
            loss_app_gen += self.L1loss(gen, gt)

            content_gen, style_gen = self.Vggloss(gen, gt) 
            loss_style_gen += style_gen
            loss_content_gen += content_gen

        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content            
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec



        loss_correctness_p, loss_regularization_p=0, 0
        loss_correctness_r, loss_regularization_r=0, 0

        for i in range(len(self.flow_fields)):
            flow_field_i = self.flow_fields[i]
            flow_p, flow_r=[],[]
            for j in range(0, len(flow_field_i), 2):
                flow_p.append(flow_field_i[j])
                flow_r.append(flow_field_i[j+1])

            correctness_r = self.Correctness(self.P_frame_step[:,i,...], self.P_reference, 
                                                    flow_r, self.opt.attn_layer)
            correctness_p = self.Correctness(self.P_frame_step[:,i,...], self.P_previous_recoder[i].detach(), 
                                                    flow_p, self.opt.attn_layer)
            loss_correctness_p += correctness_p
            loss_correctness_r += correctness_r
            loss_regularization_p += self.Regularization(flow_p)
            loss_regularization_r += self.Regularization(flow_r)


        self.loss_correctness_p = loss_correctness_p * self.opt.lambda_correct     
        self.loss_correctness_r = loss_correctness_r * self.opt.lambda_correct   
        self.loss_regularization_p = loss_regularization_p * self.opt.lambda_regularization
        self.loss_regularization_r = loss_regularization_r * self.opt.lambda_regularization


        # rec loss fake
        base_function._freeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        D_fake = self.net_D(fake)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        ##########################################################################
        base_function._freeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        # fake = [self.img_gen[i]]
        fake = []
        for frame in range(self.opt.frames_D_V-1):
            fake.append(self.img_gen[i+frame]-self.img_gen[i+frame+1])
        fake = torch.cat(fake, dim=1)
        D_fake = self.net_D_V(fake)
        self.loss_ad_gen_v = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        ##########################################################################
        
        total_loss = 0
        for name in self.loss_names:
            if name != 'dis_img_gen_v' and name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        self.update()


