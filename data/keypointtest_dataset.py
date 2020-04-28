import os.path
from data.animation_dataset import AnimationDataset
from data.keypoint_dataset import KeypointDataset
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
from util import openpose_utils
import json
import glob


class KeypointTestDataset(AnimationDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = KeypointDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--pose_extractor', type=str, default='alphapose', help='openpose | alphapose' )
        parser.add_argument('--sub_dataset', type=str, default='iper', help='iper | fashon')
        parser.add_argument('--eval_set', type=str, default='train', help='train | test')
        parser.add_argument('--start_frame', type=int, default=0)
        parser.add_argument('--total_test_frames', default=None)
        parser.set_defaults(load_size=256)
        parser.set_defaults(nThreads=1)

        return parser

    def initialize(self, opt):
        assert opt.isTrain == False
        self.opt = opt
        self.A_paths, self.B_paths_noise = self.get_paths(opt)
        self.init_frame_idx(self.A_paths)
        self.load_size = (self.opt.load_size, self.opt.load_size)

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.eval_set +'_256'
        dir_A = os.path.join(opt.dataroot, phase, 'train_A')
        dir_B_noise = os.path.join(opt.dataroot, phase, 'train_'+self.opt.pose_extractor)
        A_paths = sorted(make_grouped_dataset(dir_A)) 
        B_paths_noise = sorted(make_grouped_dataset(dir_B_noise)) 
        check_path_valid(A_paths, B_paths_noise)
        return A_paths, B_paths_noise



    def __getitem__(self, index):
        _, _, _, seq_idx = self.update_seq_idx(self.B_paths_noise, index)  
        A_paths = self.A_paths[seq_idx]
        B_paths_noise = self.B_paths_noise[seq_idx]
        gen_images = None

        n_frames_total, start_idx, t_step, org_size = self.get_video_params(self.opt, self.n_frames_total, len(B_paths_noise), self.frame_idx, A_paths)
        left_pad, current, right_pad = self.get_keypoint_param(n_frames_total, start_idx, t_step, len(B_paths_noise))
        self.org_size = (org_size[1], org_size[0])

        left,_ = self.load_keypoints(left_pad, B_paths_noise)
        current, gen_path = self.load_keypoints(current, B_paths_noise)
        right,_ = self.load_keypoints(right_pad, B_paths_noise)
        input_kp = torch.cat([left,current,right],1)
        self.frame_idx += self.opt.n_frames_pre_load_test
        
        for i in range(n_frames_total):
            A_path = A_paths[start_idx + i * t_step] 
            image = self.load_image(A_path)
            gen_images = self.concat_frame(gen_images, image)

        return_list = {'gen_keypoints': input_kp,  'current_keypoints':current, 'gen_images':gen_images,
                       'shape':org_size, 'gen_paths':gen_path, 'frame_idx':self.frame_idx, 'change_seq':self.change_seq
                       }
                
        return return_list

    def load_image(self, A_path):
        A_img = Image.open(A_path) 
        Ai = self.transform_image(A_img, self.load_size)
        return Ai

    def load_keypoints(self, keypoint_list, B_paths):
        pose_list=[]
        B_path_recoder=[]
        for index in keypoint_list:
            B_path = B_paths[index]
            B_path_recoder.append(B_path)
            B_coor = json.load(open(B_path))["people"]
            if len(B_coor)==0:
                pose = torch.zeros(17*2, 1)
            else:
                B_coor = B_coor[0]
                pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.org_size)
                pose = openpose_utils.openpose18_to_coco17(pose_dict['body'])
                pose = torch.Tensor(pose).float()
                pose = pose.view(17*2, 1)
                # normalize pose
                pose = 2*pose/self.load_size[1]-1

            pose_list.append(pose)
        B = torch.cat(pose_list, 1)
        return B, B_path_recoder 

    def get_keypoint_param(self, n_frames_total, start_idx, t_step, cur_seq_len):
        end_id = start_idx + n_frames_total * t_step
        current = [i for i in range(start_idx, end_id, t_step)]

        pad_range = self.opt.n_receptive//2
        left_most = start_idx - pad_range*t_step

        left_pad = [i for i in range(left_most, start_idx, t_step)]
        left_pad = [0 if i<0 else i for i in left_pad]

        right_most = start_idx + (n_frames_total+pad_range)*t_step
        right_pad = [i for i in range(end_id, right_most, t_step)]
        right_pad = [cur_seq_len-1 if i>cur_seq_len-1 else i for i in right_pad]
        return left_pad, current, right_pad


    def update_seq_idx(self, A_paths, index):
        self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
        if not self.change_seq and self.frame_idx + self.opt.n_frames_pre_load_test > self.frames_count[self.seq_idx]:
            self.frame_idx = self.frames_count[self.seq_idx]-self.opt.n_frames_pre_load_test
        if self.change_seq:
            self.seq_idx += 1
            self.frame_idx = self.opt.start_frame
        return None, None, None, self.seq_idx

    
    def __len__(self):
        return sum(self.frames_count)


    def name(self):
        return 'keypointDataset'