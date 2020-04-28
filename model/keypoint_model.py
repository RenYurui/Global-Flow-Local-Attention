import torch
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util, openpose_utils
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os, ntpath
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import cv2, json



class Keypoint(BaseModel):
    def name(self):
        return "Motion Extraction Net"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--netG', type=str, default='kpinput2d', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--lambda_mpjpe', type=float, default=1000, help='weight for image reconstruction loss')
        parser.add_argument('--write_image', action='store_false')

        parser.set_defaults(display_freq=1000)
        parser.set_defaults(eval_iters_freq=1000)
        parser.set_defaults(print_freq=1000)
        parser.set_defaults(save_latest_freq=1000)
        parser.set_defaults(save_iters_freq=50000)
        parser.set_defaults(write_image=False)


        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['mpjpe']
        self.model_names = ['G']
        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if len(self.gpu_ids)>0 \
            else torch.ByteTensor

        self.net_G = network.define_g(opt, filename='generator', structure_nc=opt.structure_nc, channels=256, layers=4)
        if len(opt.gpu_ids) > 1:
            self.net_G = torch.nn.DataParallel(self.net_G, device_ids=self.gpu_ids)

        self.convert2skeleton = openpose_utils.tensor2skeleton()

        if self.isTrain:
            self.L2loss = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
        else:
            self.L2loss = torch.nn.MSELoss()
            util.mkdir(os.path.join(self.opt.results_dir))

        self.setup(opt)


    def set_input(self, data):
        opt = self.opt
        if self.isTrain:
            self.input_skeleton = data['input_data'].cuda()
            self.gt_skeleton   = data['gt_data'].cuda()
            self.camera   = data['camera']
        else:
            assert self.opt.dataset_mode == 'keypointtest'
            self.input_skeleton = data['gen_keypoints'].cuda()
            self.current_skeleton = data['current_keypoints'].cuda()
            self.gen_images = data['gen_images']
            self.shape = data['shape']
            self.frame_idx = data['frame_idx']
            if self.frame_idx == self.opt.start_frame+self.opt.n_frames_pre_load:
                if data['change_seq'] and self.opt.write_image:
                    self.write2video(name_list=['skeleton_in','skeleton_out'])
                elif data['change_seq']:
                    print(self.results_dir_base)
            self.image_paths = data['gen_paths']
            self.results_dir_base = os.path.join(self.opt.results_dir,
                                                self.image_paths[0][0].split('/')[-2])                

    def update(self):
        self.output_skeleton = self.net_G(self.input_skeleton)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def test(self):
        output_skeleton = self.net_G(self.input_skeleton)
        if self.opt.write_image:
            current_map = self.convert2skeleton(self.current_skeleton, kp_form='COCO_17')
            obtained_map = self.convert2skeleton(output_skeleton, kp_form='human36m_17')
            for j in range(len(current_map)):
                short_path = ntpath.basename(self.image_paths[j][0])  
                name = os.path.splitext(short_path)[0]
                util.mkdir(self.results_dir_base)

                gt = self.gen_images[0,j*3:j*3+3,...].detach().cpu().numpy()
                gt = np.transpose(((gt+1)/2 * 255), (1,2,0))

                in_img_name = '%s_%s.%s' % (name, 'skeleton_in', 'png')
                img_path = os.path.join(self.results_dir_base, in_img_name)
                gt_ = np.copy(gt)
                gt_[current_map[j]!=0]=0
                skeleton_in = (current_map[j]+gt_).astype(np.uint8)       


                util.save_image(skeleton_in, img_path) 
                print(img_path)

                out_img_name = '%s_%s.%s' % (name, 'skeleton_out', 'png')
                img_path = os.path.join(self.results_dir_base, out_img_name)
                gt_ = np.copy(gt)
                gt_[obtained_map[j]!=0]=0                
                skeleton_out = (obtained_map[j]+gt_).astype(np.uint8)       

                util.save_image(skeleton_out, img_path)  

        output_skeleton = (output_skeleton+1)/2
        skeleton_pad = torch.ones(1,17).type_as(output_skeleton)
        for j in range(output_skeleton.shape[-1]):
            skeleton = output_skeleton[0,:,j].view(2,17)
            skeleton[0,:] = skeleton[0,:]*self.shape[-1][0]
            skeleton[1,:] = skeleton[1,:]*self.shape[0][0]
            skeleton = torch.cat((skeleton,skeleton_pad), 0)

            skeleton = skeleton[[1,0,2],:]
            skeleton = skeleton.transpose(1,0).contiguous().view(-1).detach().cpu().numpy()
            people = dict()
            people['pose_keypoints_2d']=skeleton.tolist()
            output_dict=dict()
            output_dict["version"]="Video to Pose 2D"
            output_dict["people"]=[people]
            
            short_path = ntpath.basename(self.image_paths[j][0])  
            name = os.path.splitext(short_path)[0]
            util.mkdir(self.results_dir_base)
            name = '%s.%s' % (name, 'json')
            name = os.path.join(self.results_dir_base, name)
            with open(name, 'w') as f:
                json.dump(output_dict, f)



    def write2video(self, name_list):
        images=[]
        for name in name_list:
            images.append(sorted(glob.glob(self.results_dir_base+'/*_'+name+'.png')))

        image_array=[]
        for i in range(len(images[0])):
            cat_im=None
            for image_list in images:
                im = cv2.imread(image_list[i])
                if cat_im is not None:
                    cat_im = np.concatenate((cat_im, im), axis=1)
                else:
                    cat_im = im
            image_array.append(cat_im) 

        res=''
        for name in name_list:
            res += (name +'_')
        out_name = self.results_dir_base+'_'+res+'.mp4' 
        print('write video %s'%out_name)  
        height, width, layers = cat_im.shape
        size = (width,height)
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
        for i in range(len(image_array)):
            out.write(image_array[i])
        out.release()

        

    def backward_G(self):
        """Calculate training loss for the generator"""
        self.loss_mpjpe = self.L2loss(self.output_skeleton, self.gt_skeleton)

        total_loss = 0
        total_loss = self.loss_mpjpe
        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        self.update()


