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
import warnings

class AnimationDataset(BaseDataset):
    '''
    The dataset for animation tasks.
    For Example: 
        Face Image Animation 
        Pose-Guided Person Image Animation
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--n_frames_total', type=int, default=30, help='the overall number of frames in a sequence to train with')  
        parser.add_argument('--max_frames_per_gpu', type=int, default=6, help='number of frames to load into one GPU at a time')
        parser.add_argument('--n_frames_pre_load_test', type=int, default=1, help='number of frames to load every time when test')
        parser.add_argument('--total_test_frames', type=int, default=300, help='the overall number of frames to load every sequence when test')
        parser.add_argument('--max_t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        parser.add_argument('--debug', action='store_true', help='debuge mode') 

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.A_paths, self.B_paths, self.C_paths = self.get_paths(opt)
        self.init_frame_idx(self.A_paths)
        self.load_size = (self.opt.load_size, self.opt.load_size)


    def init_frame_idx(self, A_paths):
        self.n_of_seqs = min(len(A_paths), self.opt.max_dataset_size)         # number of sequences to train
        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence

        test_total, test_pre_load = self.opt.total_test_frames, self.opt.n_frames_pre_load_test
        if test_total is not None and test_total%test_pre_load != 0:
            n_test_load = test_total // test_pre_load
            test_total = test_pre_load * n_test_load
            warnings.warn("Cannot Load %d frames for each sequence as it is not divisible."
                          "Change the 'total_test_frames' to %d"%(self.opt.total_test_frames, test_total) )
            self.opt.total_test_frames = test_total

        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            self.frames_count.append(len(path))
            if self.opt.total_test_frames is not None:
                assert self.opt.total_test_frames<=len(path), "Sequence %s does not have enough frames"%(os.path.dirname(path[0]))
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else self.opt.n_frames_pre_load_test


    def update_seq_idx(self, A_paths, index):
        if self.opt.isTrain:
            seq_idx = index % self.n_of_seqs            
            return None, None, None, seq_idx
        else:
            if self.opt.total_test_frames is not None:
                self.change_seq = self.frame_idx >= (self.opt.total_test_frames + self.opt.start_frame) 
            else:
                self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = self.opt.start_frame
            return None, None, None, self.seq_idx

    def get_video_params(self, opt, n_frames_total, cur_seq_len, frame_idx, img_paths):
        if opt.isTrain:        
            n_frames_total = min(n_frames_total, cur_seq_len)          # number of frames to load every dataset.item() 
            n_frames_per_load = opt.max_frames_per_gpu                 # number of frames to load into GPUs at one time 
            n_frames_per_load = min(n_frames_total, n_frames_per_load)
            n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs         
            n_frames_total = n_frames_per_load * n_loadings            # rounded overall number of frames to read from the sequence
            
            max_t_step = min(opt.max_t_step, cur_seq_len//n_frames_total)
            t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames
            offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame        

            start_idx = np.random.randint(offset_max)                 # offset for the first frame to load
            if opt.debug:
                print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                    % (n_frames_total, start_idx, t_step))
        else:
            start_idx = frame_idx
            t_step = 1  
            if opt.debug:
                print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                    % (n_frames_total, start_idx, t_step))  

        img = Image.open(img_paths[start_idx])
        img_size = img.size
        return n_frames_total, start_idx, t_step, img_size



    def transform_image(self, image, resize_param, method=Image.BICUBIC, affine=None, normalize=True, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = None if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image                 



    def concat_frame(self, A, Ai):
        if A is None:
            A = Ai
        else:
            A = torch.cat([A, Ai])
        return A


    def __len__(self):
        if self.opt.isTrain:
            return len(self.A_paths)
        else:
            return sum(self.frames_count) // self.opt.n_frames_pre_load_test


    def name(self):
        return 'FaceDataset'