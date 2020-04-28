import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import interpPoints, drawEdge
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import cv2
from skimage import feature
from util import human36m
import json
import glob


class KeypointDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        parser.add_argument('--n_frames_pre_load', type=int, default=24, help='number of output keypoints pre load')
        parser.add_argument('--n_frames_pre_load_test', type=int, default=24, help='number of output keypoints pre load')
        parser.add_argument('--n_receptive', type=int, default=81, help='The receptive fields of the network')
        parser.add_argument('--gt_path', type=str, default='./dataset/human36m/data_2d_h36m_gt.npz')
        parser.add_argument('--input_path', type=str, default='./dataset/human36m/data_2d_h36m_detectron_pt_coco.npz')
        parser.add_argument('--debug', action='store_true', help='debuge mode')  

        parser.set_defaults(debug=False)
        parser.set_defaults(structure_nc=17)
        parser.set_defaults(batchSize=32)

        if not is_train:
            parser.set_defaults(batchSize=1)

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.gt_data, self.gt_data_type = human36m.init_position_data(self.opt.gt_path)
        self.input_data, self.input_data_type = human36m.init_position_data(self.opt.input_path)
        assert self.input_data_type == '2d', self.gt_data_type == '2d'
        self.cameras = human36m.init_cameras_param()
        if self.opt.isTrain:
            self.data_list = human36m.init_video_list(self.opt.input_path, self.opt.gt_path, ['S9', 'S11'])
            self.data_list = human36m.check_dataset(self.gt_data, self.input_data, self.data_list, self.gt_data_type)
        else:
            self.data_list = human36m.init_video_list(self.opt.input_path, self.opt.gt_path, ['S1', 'S5', 'S6', 'S7', 'S8'])
            self.data_list = human36m.check_dataset(self.gt_data, self.input_data, self.data_list, self.gt_data_type)

    def random_frames(self, input_data):
        cur_seq_len = input_data.shape[0]
        n_frames_pre_load = self.opt.n_frames_pre_load
        t_step = min(self.opt.t_step, cur_seq_len//n_frames_pre_load) # frame inperpolation
        offset_max = max(1, cur_seq_len-(n_frames_pre_load-1)*t_step)  # maximum possible index for the first frame        

        start_idx = np.random.randint(offset_max)                 # offset for the first frame to load


        gt_keypoint_list=[]
        for i in range(0, n_frames_pre_load):
            index = start_idx+i*t_step
            gt_keypoint_list.append(index)  


        input_keypoint_list = []
        for i in range(-self.opt.n_receptive // 2+1, n_frames_pre_load+self.opt.n_receptive//2):
            index = start_idx+i*t_step
            index = 0 if index < 0 else index
            index = cur_seq_len-1 if index > cur_seq_len-1 else index
            input_keypoint_list.append(index) 

        if self.opt.debug:
            print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                % (n_frames_pre_load, start_idx, t_step))       
            print('input index',input_keypoint_list)     
            print('gt index',   gt_keypoint_list)     
        return  gt_keypoint_list, input_keypoint_list



    def __getitem__(self, index):
        seq_idx = self.update_seq_idx(index)  
        seq_name = self.data_list[seq_idx]

        input_data = self.input_data[seq_name['subject']][seq_name['action']][seq_name['camera']]
        input_data = input_data[:,:,:-1] if 'detectron_pt_coco' in self.opt.input_path else input_data
        camera = self.cameras[seq_name['subject']][int(seq_name['camera'])]
        gt_data = self.obtain_gt_data(seq_name, camera)
        if input_data.shape[0]> gt_data.shape[0]:
            input_data = input_data[:gt_data.shape[0], ...]
        else:
            gt_data = gt_data[:input_data.shape[0],...]

        if self.opt.isTrain:
            gt_keypoint_list, input_keypoint_list = self.random_frames(input_data)

            input_data = torch.tensor(input_data)[input_keypoint_list,...]
            input_data = self.data_normalization(input_data, self.input_data_type, camera)

            gt_data = torch.tensor(gt_data)[gt_keypoint_list,...]
            gt_data = self.data_normalization(gt_data, self.gt_data_type, camera)
            gt_data, input_data = self.random_transformation(gt_data, input_data)

            return_list = {'input_data': input_data, 'gt_data': gt_data, 'camera':camera}
        else:
            input_data = torch.tensor(input_data)
            gt_data    = torch.tensor(gt_data)
            gt_data = self.data_normalization(gt_data, self.gt_data_type, camera)
            input_data = self.data_normalization(input_data, self.input_data_type, camera)

            return_list = {'input_data': input_data, 'gt_data': gt_data, 'camera':camera, 
                           'out_path':seq_name}
        return return_list

    def obtain_gt_data(self, seq_name, cam):
        gt_data = self.gt_data[seq_name['subject']][seq_name['action']][seq_name['camera']]
        gt_data = gt_data[:,:,:2] # remove possible depth (x,y,depth)

        return gt_data

    def data_normalization(self, data, data_type, cam):
        if data_type == '2d':
            data = data/cam['res_w']*2-1
            data = data.permute(0,2,1)[:,[1,0],:]
            data = data.view(data.shape[0],-1).permute(1,0)
        else:
            data = data.permute(0,2,1)[:,[1,0,2],:]
            data[:,:,1:] -= data[:,:,:1] # shape and trance
            # Bring the skeleton to 17 joints instead of the original 32
            data = data[...,[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]]
            data = data.view(data.shape[0],-1).permute(1,0)
        return data

    def random_transformation(self, gt_data, input_data):
        if self.gt_data_type == '2d':
            scale = 0.5*np.random.random()+1      
            [delta_y, delta_x]=[np.random.random()*0.2-0.1, np.random.random()*0.2-0.1]
            delta = torch.tensor([delta_y,delta_x]).view(1,2,1)

            input_delta = delta.expand(input_data.shape[-1],-1,17).contiguous().view(input_data.shape[-1],-1).permute(1,0)
            input_data = input_data*scale+input_delta

            gt_delta = delta.expand(gt_data.shape[-1],-1,17).contiguous().view(gt_data.shape[-1],-1).permute(1,0)
            gt_data = gt_data*scale+gt_delta
        else:
            pass
        return gt_data, input_data

    def update_seq_idx(self, index):
        seq_idx = index % len(self.data_list)            
        return seq_idx

    
    def __len__(self):
        return len(self.data_list)


    def name(self):
        return 'keypointDataset'