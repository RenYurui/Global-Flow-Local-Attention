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


class PoseFlowNet(BaseModel):
    def name(self):
        return "Pre-train flow estimator for human pose image generation"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--netG', type=str, default='poseflownet', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...", help="The number layers away from output layer") 
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", help="Kernel Size of Local Attention Block")

        parser.add_argument('--lambda_correct', type=float, default=20.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.01, help='weight for Regularization loss')
        parser.add_argument('--use_spect_g', action='store_false')
        parser.set_defaults(use_spect_g=False)
        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['correctness', 'regularization']
        self.visual_names = ['input_P1','input_P2', 'warp', 'flow_fields',
                            'masks','input_BP1', 'input_BP2']
        self.model_names = ['G']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        self.net_G = network.define_g(opt, structure_nc=opt.structure_nc, ngf=32, img_f=256, 
                                       layers=5, norm='instance', activation='LeakyReLU', 
                                       attn_layer=self.opt.attn_layer, use_spect=opt.use_spect_g,
                                       )
        self.flow2color = util.flow2color()

        if self.isTrain:
            # define the loss functions
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)
            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                                lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
        # load the pretrained model and schedulers
        self.setup(opt)


    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], async=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], async=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], async=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], async=True)  

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_2_' + input['P2_path'][i])


    def forward(self):
        """Run forward processing to get the inputs"""
        self.flow_fields, self.masks = self.net_G(self.input_P1, self.input_BP1, self.input_BP2)
        self.warp  = self.visi(self.flow_fields[-1])

    def visi(self, flow_field):
        [b,_,h,w] = flow_field.size()

        source_copy = torch.nn.functional.interpolate(self.input_P1, (h,w))

        x = torch.arange(w).view(1, -1).expand(h, -1).float()
        y = torch.arange(h).view(-1, 1).expand(-1, w).float()
        x = 2*x/(w-1)-1
        y = 2*y/(h-1)-1
        grid = torch.stack([x,y], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        flow_x = (2*flow_field[:,0,:,:]/(w-1)).view(b,1,h,w)
        flow_y = (2*flow_field[:,1,:,:]/(h-1)).view(b,1,h,w)
        flow = torch.cat((flow_x,flow_y), 1)

        grid = (grid+flow).permute(0, 2, 3, 1)
        warp = torch.nn.functional.grid_sample(source_copy, grid)
        return  warp


    def backward_G(self):
        """Calculate training loss for the generator"""
        loss_correctness = self.Correctness(self.input_P2, self.input_P1, self.flow_fields, self.opt.attn_layer)
        self.loss_correctness = loss_correctness * self.opt.lambda_correct

        loss_regularization = self.Regularization(self.flow_fields)
        self.loss_regularization = loss_regularization * self.opt.lambda_regularization

        total_loss = 0
        for name in self.loss_names:
            total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update netowrk weights"""
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
