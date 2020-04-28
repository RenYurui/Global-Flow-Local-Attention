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


class ShapeNet(BaseModel):
    def name(self):
        return "ShapeNet Novel View Synthesis"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='shapenet', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for regularization loss')

        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen','correctness_gen', 'content_gen', 'style_gen', 'regularization',
                           'ad_gen', 'dis_img_gen']


        self.visual_names = ['input_P1','input_P2', 'img_gen', 'flow_fields','masks']
        self.model_names = ['G','D']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)

        self.flow2color = util.flow2color()
        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)

        if self.isTrain:
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
                                filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)

        self.setup(opt)


    def set_input(self, input):
        self.input = input
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        if len(self.gpu_ids) > 0:
            if self.opt.isTrain:
                self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)
                self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
                self.input_P2 = input_P2.cuda(self.gpu_ids[0], async=True)
                self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], async=True)        
            else:
                self.input_P1  = input_P1.cuda(self.gpu_ids[0], async=True)
                self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
                self.input_P2  = [item.cuda(self.gpu_ids[0], async=True) for item in input_P2]
                self.input_BP2 = [item.cuda(self.gpu_ids[0], async=True) for item in input_BP2]

        if self.opt.isTrain:
            self.input_BP1 = self.obtain_shape_net_semantic(self.input_BP1)
            self.input_BP2 = self.obtain_shape_net_semantic(self.input_BP2)

            self.image_paths=[]
            for i in range(self.input_P1.size(0)):
                self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])
        else: 
            self.image_paths=[]
            for i in range(self.input_P1.size(0)):
                self.image_paths.append(input['P1_path'][i].split('_')[0])

    def obtain_shape_net_semantic(self, inputs):
        inputs_h = inputs[:,0,:,:].unsqueeze(1)/2
        inputs_v = inputs[:,1,:,:].unsqueeze(1)/10
        semanctic_h = self.label2semantic(inputs_h, self.opt.label_nc_h)
        semanctic_v = self.label2semantic(inputs_v, self.opt.label_nc_v)
        return torch.cat((semanctic_h, semanctic_v), 1)

    def label2semantic(self, label, nc):
        bs, _, h, w = label.size()
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        semantics = input_label.scatter_(1, label, 1.0)
        return semantics

    def test(self):
        """Forward function used in test time"""
        for j, input_BP2 in enumerate(self.input_BP2):
            input_BP2 = self.obtain_shape_net_semantic(input_BP2)
            input_BP1 = self.obtain_shape_net_semantic(self.input_BP1)
            img_gen, flow_fields, masks = self.net_G(self.input_P1, input_BP1, input_BP2)

            for i in range(self.input_P1.size(0)):
                util.mkdir(self.opt.results_dir)
                img_path = os.path.join(self.opt.results_dir, self.image_paths[i])
                util.mkdir(img_path)
                source_name = os.path.join(img_path, 'source.png')
                img_numpy = util.tensor2im(self.input_P1[i])
                util.save_image(img_numpy, source_name) 

                gt_name = os.path.join(img_path, 'gt_' + str(j).zfill(4) + '.png')
                img_numpy = util.tensor2im(self.input_P2[j][i])
                util.save_image(img_numpy, gt_name)                 

                gen_name = os.path.join(img_path, 'result_' + str(j).zfill(4) + '.png')
                img_numpy = util.tensor2im(img_gen[i])
                util.save_image(img_numpy, gen_name) 
                print(gen_name)                
               
    def forward(self):
        source_list=[]
        self.img_gen, self.flow_fields, self.masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)

    def backward_D_basic(self, netD, real, fake):
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        base_function._unfreeze(self.net_D)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)

    def backward_G(self):
        loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
        loss_correctness_gen = self.Correctness(self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness_gen = loss_correctness_gen * self.opt.lambda_correct        
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

        base_function._freeze(self.net_D)
        D_fake = self.net_D(self.img_gen)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        loss_regularization = self.Regularization(self.flow_fields)
        self.loss_regularization = loss_regularization * self.opt.lambda_regularization

        loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2) 
        self.loss_style_gen = loss_style_gen*self.opt.lambda_style
        self.loss_content_gen = loss_content_gen*self.opt.lambda_content

        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img_rec' and name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()



