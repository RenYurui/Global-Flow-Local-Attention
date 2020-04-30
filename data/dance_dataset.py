import os.path
from data.animation_dataset import AnimationDataset
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
from util import pose_utils, openpose_utils
import json
import glob
from tqdm import tqdm
import ast


class DanceDataset(AnimationDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = AnimationDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--no_bone_map', action='store_true', help='do *not* use bone RGB image as input')
        parser.add_argument('--sub_dataset', type=str, default='iper', help='iper | fashon')
        parser.add_argument('--use_kp', action='store_true', help='whether load key point joints')
        parser.add_argument('--use_mask', action='store_true', help='whether load mask to split background and foreground')

        if is_train:
            parser.set_defaults(load_size=256)
            parser.set_defaults(batchSize=2)
            parser.set_defaults(angle=True)
            parser.set_defaults(shift=True)
            parser.set_defaults(scale=True)

            parser.set_defaults(use_kp=False)
            parser.set_defaults(total_test_frames=None)
        else:
            parser.add_argument('--start_frame', type=int, default=0, help='frame index to start inference on')        
            parser.add_argument('--test_list', type=str, default=None, help='image list for test')        
            parser.add_argument('--cross_eval', action='store_true', help='use cross evaluation or not')        


            parser.set_defaults(load_size=256)
            parser.set_defaults(batchSize=1)
            parser.set_defaults(angle=False)
            parser.set_defaults(shift=False)
            parser.set_defaults(scale=False)

            parser.set_defaults(use_kp=True)
            parser.set_defaults(total_test_frames=None)
            parser.set_defaults(test_list='./test_list.csv')
            parser.set_defaults(cross_eval=False)  
            # parser.set_defaults(test_list=None)
            # parser.set_defaults(cross_eval=False)  
            parser.set_defaults(n_frames_pre_load_test=6)
            parser.set_defaults(nThreads=1)



        parser.set_defaults(image_nc=3)
        parser.set_defaults(mask_nc=1)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(no_bone_map=False)
        parser.set_defaults(debug=False)
        parser.set_defaults(structure_nc=17+3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        # use mask to train iper images 
        self.opt.use_mask = True if self.opt.sub_dataset == 'iper' and self.opt.isTrain else False
        self.A_paths, self.B_paths_clean, self.B_paths_noise, self.C_paths = self.get_paths(opt)
        self.init_frame_idx([seq['gen'] for seq in self.A_paths])
        self.load_size = (self.opt.load_size, self.opt.load_size)
        self.init_affine_param()


    def init_affine_param(self):
        if self.opt.sub_dataset == 'fashion':
            if self.opt.angle == True:
                self.opt.angle = (-5, 5) # random angle in range (-5, 5)           
            if self.opt.shift == True:
                self.opt.shift = (20, 3) # maximum shift_x=20, shift_y=3
            if self.opt.scale == True:
                self.opt.scale = (0.98, 1.02) # random scale in range (0.98, 1.02)  
        elif self.opt.sub_dataset == 'iper':
            if self.opt.angle == True:
                self.opt.angle = (-5, 5)            
            if self.opt.shift == True:
                self.opt.shift = False 
            if self.opt.scale == True:
                self.opt.scale = False

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        phase_dir = phase+'_256'
        self.phase_dir = phase_dir
        dir_A = os.path.join(opt.dataroot, phase_dir, 'train_A')
        dir_B_clean = os.path.join(opt.dataroot, phase_dir, 'train_'+'video2d')
        dir_B_noise = os.path.join(opt.dataroot, phase_dir, 'train_'+'alphapose')

        A_paths = sorted(make_grouped_dataset(dir_A))
        B_paths_clean = sorted(make_grouped_dataset(dir_B_clean)) 
        B_paths_noise = sorted(make_grouped_dataset(dir_B_noise)) 
        check_path_valid(A_paths, B_paths_clean)
        check_path_valid(A_paths, B_paths_noise)

        if phase == 'train':
            if self.opt.use_mask:
                dir_C = os.path.join(opt.dataroot, phase_dir, 'train_C')
                C_paths = sorted(make_grouped_dataset(dir_C))
                check_path_valid(A_paths, C_paths)
                C_paths = self.split_ref_gen(C_paths, self.opt.sub_dataset)
            else:
                C_paths = None

            A_paths = self.split_ref_gen(A_paths, self.opt.sub_dataset)
            B_paths_clean = self.split_ref_gen(B_paths_clean, self.opt.sub_dataset)
            B_paths_noise = self.split_ref_gen(B_paths_noise, self.opt.sub_dataset)            
        else:
            assert self.opt.use_mask == False
            C_paths = None
            if opt.test_list is not None:
                scv_path = os.path.join(opt.dataroot, opt.test_list)
                file = pd.read_csv(scv_path)
                A_paths = file['A_paths'].map(ast.literal_eval)
                B_paths_noise = file['B_paths_noise'].map(ast.literal_eval)
                B_paths_clean = file['B_paths_clean'].map(ast.literal_eval)
            else:
                A_paths = self.split_ref_gen(A_paths, self.opt.sub_dataset)
                B_paths_clean = self.split_ref_gen(B_paths_clean, self.opt.sub_dataset)
                B_paths_noise = self.split_ref_gen(B_paths_noise, self.opt.sub_dataset)   
            A_paths, B_paths_clean, B_paths_noise = self.pad_for_latest_frames(A_paths, B_paths_clean, B_paths_noise)
        return A_paths, B_paths_clean, B_paths_noise, C_paths  

    def pad_for_latest_frames(self, A_paths, B_paths_clean, B_paths_noise):
        in_list = [A_paths, B_paths_clean, B_paths_noise]
        out_list=[]
        for paths in in_list:
            padded_paths=[]
            for path_dict in paths:
                org_len_gen = len(path_dict['gen'])
                if org_len_gen%self.opt.n_frames_pre_load_test == 0:
                    pass
                else:
                    pad = self.opt.n_frames_pre_load_test - org_len_gen%self.opt.n_frames_pre_load_test
                    pad_files = [path_dict['gen'][-1]]*pad
                    path_dict['gen'].extend(pad_files)
                padded_paths.append(path_dict)
            out_list.append(padded_paths) 
        [A_paths, B_paths_clean, B_paths_noise] = out_list
        return A_paths, B_paths_clean, B_paths_noise

        
    def split_ref_gen(self, paths, sub_dataset):
        out_paths=[]
        for path in paths:
            dic=dict()
            dic['gen'] = path
            if not self.opt.isTrain and self.opt.cross_eval:
                index = np.random.randint(len(paths))
                dic['ref'] = paths[index][0:20]
            else:
                dic['ref'] = path[0:20]
            out_paths.append(dic)
        return out_paths

    def __getitem__(self, index):
        _, _, _, seq_idx = self.update_seq_idx(self.A_paths, index)  
        gen_images, gen_kps_noise, gen_kps_clean, gen_masks, gen_skeleton = None, None, None, None, None
        A_paths = self.A_paths[seq_idx]['gen'] # image path
        C_paths = self.C_paths[seq_idx]['gen'] if self.opt.use_mask else None # mask path
        B_paths_clean = self.B_paths_clean[seq_idx]['gen'] # key point path
        B_paths_noise = self.B_paths_noise[seq_idx]['gen'] # noise key point path

        self.affine_param = self.getRandomAffineParam()
        n_frames_total, start_idx, t_step, org_size = self.get_video_params(self.opt, self.n_frames_total, len(A_paths), self.frame_idx, A_paths)
        self.org_size = (org_size[1], org_size[0]) 
        gen_paths=[]

        for i in range(n_frames_total):
            # load image
            A_path = A_paths[start_idx + i * t_step] 
            image = self.load_image(A_path)
            gen_images = self.concat_frame(gen_images, image)
            gen_paths.append(A_path)

            # load skeleton maps (b,c,h,w)  
            B_path = B_paths_clean[start_idx + i * t_step] 
            skeleton = self.load_skeleton(B_path)
            gen_skeleton = self.concat_frame(gen_skeleton, skeleton)

            if self.opt.use_kp:
                # load key point joints
                B_path = B_paths_clean[start_idx + i * t_step] 
                kp = self.load_keypoint(B_path, is_clean_pose=True) 
                gen_kps_clean = self.concat_kps(gen_kps_clean, kp)
                B_path = B_paths_noise[start_idx + i * t_step] 
                kp = self.load_keypoint(B_path, is_clean_pose=False) 
                gen_kps_noise = self.concat_kps(gen_kps_noise, kp)                

            if self.opt.use_mask:
                # load masks
                C_path = C_paths[start_idx + i * t_step] 
                mask = self.load_mask(C_path) 
                gen_masks = self.concat_frame(gen_masks, mask)
        ref_image, ref_skeleton, ref_path = self.gen_ref_image(seq_idx)

        if not self.opt.isTrain:
            self.frame_idx += self.opt.n_frames_pre_load_test
            if self.opt.total_test_frames is not None:
                seq_total_frame = self.opt.total_test_frames
            else:
                seq_total_frame = self.frames_count[self.seq_idx]
            change_seq = self.frame_idx >= seq_total_frame            
        else:
            change_seq = None
        return_list = {'gen_images': gen_images,  'gen_masks':gen_masks, 
                       'gen_skeleton':gen_skeleton,
                       'ref_image':ref_image, 'ref_skeleton':ref_skeleton,
                       'gen_kps_clean':gen_kps_clean, 'gen_kps_noise':gen_kps_noise,
                       'gen_paths': gen_paths, 'ref_path':ref_path,
                       'frame_idx':self.frame_idx, 'change_seq':change_seq
                       }
        return_list = {k: v for k, v in return_list.items() if v is not None}
        return return_list

    def concat_kps(self, kps, kp):
        if kps is None:
            kps = kp
        else:
            kps = torch.cat((kps, kp), 1)
        return kps


    def gen_ref_image(self, seq_idx):
        if self.opt.sub_dataset == 'fashion':
            # renew the affine augmentation params 
            # since we do not need to align the background
            self.affine_param = self.getRandomAffineParam()

        ref_A_paths = self.A_paths[seq_idx]['ref']
        ref_B_paths = self.B_paths_noise[seq_idx]['ref']
        index = np.random.randint(0, len(ref_A_paths))
        A_path = ref_A_paths[index]
        B_path = ref_B_paths[index]
        ref_image = self.load_image(A_path)
        ref_skeleton = self.load_skeleton(B_path, False)
        ref_path = A_path
        return ref_image, ref_skeleton, ref_path

    def load_image(self, A_path):
        A_img = Image.open(A_path) 
        # padding white color after affine transformation  
        fillWhiteColor = True if self.opt.sub_dataset=='fashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=self.affine_param, fillWhiteColor=fillWhiteColor)
        return Ai


    def load_skeleton(self, B_path, is_clean_pose=True):
        B_coor = json.load(open(B_path))["people"]
        if len(B_coor)==0:
            pose = torch.zeros(self.opt.structure_nc, self.load_size[0], self.load_size[1])
        else:
            B_coor = B_coor[0]
            pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.org_size, affine=self.affine_param)
            pose_body = pose_dict['body']
            if not is_clean_pose:
                pose_body = openpose_utils.openpose18_to_coco17(pose_body)

            pose_numpy = openpose_utils.obtain_map(pose_body, self.load_size) 
            pose = np.transpose(pose_numpy,(2, 0, 1))
            pose = torch.Tensor(pose)
            Bi = pose
            if not self.opt.no_bone_map:
                color = np.zeros(shape=self.load_size + (3, ), dtype=np.uint8)
                LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
                color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
                color = np.transpose(color,(2,0,1))
                color = torch.Tensor(color)
                Bi = torch.cat((Bi, color), dim=0)
        return Bi

    def load_mask(self, C_path):
        C_mask = Image.open(C_path)
        Ci = self.transform_image(C_mask, self.load_size, normalize=False, affine=self.affine_param)
        return Ci

    def getRandomAffineParam(self):
        if not self.opt.angle and not self.opt.scale and not self.opt.shift:
            affine_param = None
            return affine_param
        else:
            affine_param=dict()
            affine_param['angle'] = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1]) if self.opt.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1]) if self.opt.scale is not False else 1
            shift_x = np.random.uniform(low=-self.opt.shift[0], high=self.opt.shift[0]) if self.opt.shift is not False else 0
            shift_y = np.random.uniform(low=-self.opt.shift[1], high=self.opt.shift[1]) if self.opt.shift is not False else 0
            affine_param['shift']=(shift_x, shift_y)

            return affine_param

        
    def name(self):
        return 'DanceDataset'


    def load_keypoint(self, B_path, is_clean_pose):
        B_coor = json.load(open(B_path))["people"]
        if len(B_coor)==0:
            pose = torch.zeros(17*2, 1)
        else:
            B_coor = B_coor[0]
            pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.org_size, affine=self.affine_param)
            if not is_clean_pose:
                pose = openpose_utils.openpose18_to_coco17(pose_dict['body'])
            else:
                pose = pose_dict['body']
            pose = torch.Tensor(pose).float()
            pose = pose.view(17*2, 1)
            pose = 2*pose/self.load_size[1]-1
        return pose



         