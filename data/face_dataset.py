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


class FaceDataset(AnimationDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = AnimationDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--no_canny_edge', action='store_true', help='do *not* use canny edge as input')
        parser.add_argument('--no_dist_map', action='store_true', help='do *not* use distance transform map as input')

        if is_train:
            parser.set_defaults(load_size=256)
            parser.set_defaults(batchSize=2)
            parser.set_defaults(total_test_frames=None)
        else:
            parser.set_defaults(load_size=256)
            parser.set_defaults(batchSize=1)
            parser.add_argument('--start_frame', type=int, default=0, help='frame index to start inference on')        
            parser.set_defaults(total_test_frames=None)
            parser.set_defaults(n_frames_pre_load_test=6)
            parser.set_defaults(nThreads=1)

        parser.set_defaults(structure_nc=16)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)

        parser.set_defaults(n_frames_total=30)
        parser.set_defaults(max_frames_per_gpu=6)
        parser.set_defaults(max_t_step=1)

        parser.set_defaults(debug=False)


        return parser


    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        dir_A = os.path.join(opt.dataroot, phase + '_keypoints')
        dir_B = os.path.join(opt.dataroot, phase + '_data')

        A_paths = sorted(make_grouped_dataset(dir_A))
        B_paths = sorted(make_grouped_dataset(dir_B)) 
        check_path_valid(A_paths, B_paths)
        if self.opt.phase == 'test' or self.opt.phase == 'val':
            A_paths, B_paths = self.pad_for_latest_frames(A_paths, B_paths)
        return A_paths, B_paths, None

    def pad_for_latest_frames(self, A_paths, B_paths):
        in_list = [A_paths, B_paths]
        out_list=[]
        for paths in in_list:
            padded_paths=[]
            for path in paths:
                org_len_gen = len(path)
                if org_len_gen%self.opt.n_frames_pre_load_test == 0:
                    pass
                else:
                    pad = self.opt.n_frames_pre_load_test - org_len_gen%self.opt.n_frames_pre_load_test
                    pad_files = [path[-1]]*pad
                    path.extend(pad_files)
                padded_paths.append(path)
            out_list.append(padded_paths) 
        [A_paths, B_paths] = out_list
        return A_paths, B_paths

    def __getitem__(self, index):
        A, B, _, seq_idx = self.update_seq_idx(self.A_paths, index)        
        A_paths = self.A_paths[seq_idx]
        B_paths = self.B_paths[seq_idx]
        n_frames_total, start_idx, t_step, B_size = self.get_video_params(self.opt, self.n_frames_total, len(A_paths), self.frame_idx, B_paths)
               
        transform_scaleA = self.get_transform(self.opt,  method=Image.BILINEAR, normalize=False)
        transform_label = self.get_transform(self.opt,  method=Image.NEAREST, normalize=False)
        transform_scaleB = self.get_transform(self.opt)
        
        # read in images       
        image_path=[] 
        frame_range = list(range(n_frames_total))  
        for i in frame_range:
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]                    
            B_img = Image.open(B_path)
            Ai, Li = self.get_face_image(A_path, transform_scaleA, transform_label, B_size, B_img)
            Ai  = torch.cat([Ai, Li])
            Bi = transform_scaleB(B_img)
            A = self.concat_frame(A, Ai)
            B = self.concat_frame(B, Bi)
            image_path.append(B_path)
        
        if not self.opt.isTrain:
            self.frame_idx += self.opt.n_frames_pre_load_test
            
            if self.opt.total_test_frames is not None:
                seq_total_frame = self.opt.total_test_frames
            else:
                seq_total_frame = self.frames_count[self.seq_idx]
            change_seq = self.frame_idx >= seq_total_frame

        change_seq = False if self.opt.isTrain else change_seq

        return_list = {'BP': A, 'P': B, 'BP_path': A_path, 
                        'P_path': image_path,'change_seq': change_seq, 'frame_idx':self.frame_idx}
                
        return return_list



    def get_transform(self, opt, method=Image.BICUBIC, normalize=True, toTensor=True):
        transform_list = []
        ### resize input image
        if isinstance(opt.load_size, int):
            self.osize = [opt.load_size, opt.load_size]
        else:
            self.osize = opt.load_size
        transform_list.append(transforms.Resize(self.osize, method))   

        if toTensor:
            transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)                
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def get_face_image(self, A_path, transform_A, transform_L, size, img):
        # read face keypoints from path
        keypoints, part_list, part_labels = self.read_keypoints(A_path, size)

        # draw edges and possibly add distance transform maps
        add_dist_map = not self.opt.no_dist_map
        im_edges, dist_tensor = self.draw_face_edges(keypoints, part_list, transform_A, size, add_dist_map, self.osize)
        
        # canny edge for background
        if not self.opt.no_canny_edge:
            img = F.resize(img, self.osize)
            part_labels = Image.fromarray(part_labels)
            part_labels = F.resize(part_labels, self.osize, interpolation=Image.NEAREST)
            edges = feature.canny(np.array(img.convert('L')))        
            edges = edges * (np.array(part_labels) == 0)  # remove edges within face
            im_edges += (edges * 255).astype(np.uint8)
        edge_tensor = transform_A(Image.fromarray(im_edges))

        # final input tensor
        input_tensor = torch.cat([edge_tensor, dist_tensor]) if add_dist_map else edge_tensor
        label_tensor = transform_L(part_labels) * 255.0
        return input_tensor, label_tensor

    def read_keypoints(self, A_path, size):        
        # mapping from keypoints to face part 
        part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                     [range(60, 65), [64,65,66,67,60]]                 # tongue
                    ]
        label_list = [1, 2, 2, 3, 4, 4, 5, 6] # labeling for different facial parts        
        keypoints = np.loadtxt(A_path, delimiter=',')
        
        # add upper half face by symmetry
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0,1] + pts[-1,1]) / 2
        upper_pts = pts[1:-1,:].copy()
        upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1,:]))  

        # label map for facial part
        w, h = size
        part_labels = np.zeros((h, w), np.uint8)
        for p, edge_list in enumerate(part_list):                
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p]) 

        return keypoints, part_list, part_labels

    def draw_face_edges(self, keypoints, part_list, transform_A, size, add_dist_map, outsize=(256,256)):
        w, h = size
        w_o, h_o =  outsize
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        
        # edge map for face region from keypoints
        im_edges = np.zeros((h_o, w_o), np.uint8) # edge map for all edges
        dist_tensor = 0
        e = 1                
        for edge_list in part_list:
            for edge in edge_list:
                im_edge = np.zeros((h_o, w_o), np.uint8) # edge map for the current edge
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0].astype(np.float32)/ w * w_o
                    y = keypoints[sub_edge, 1].astype(np.float32)/ h * h_o
                                    
                    curve_x, curve_y = interpPoints(x.astype(np.int), y.astype(np.int)) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y, bw=0)
                    if add_dist_map:
                        drawEdge(im_edge, curve_x, curve_y, bw=0)
                                
                if add_dist_map: # add distance transform map on each facial part
                    im_edge_tr = Image.fromarray(im_edge).resize(self.osize, Image.NEAREST)
                    im_edge_tr = np.asarray(im_edge_tr)
                    im_dist = cv2.distanceTransform(255-im_edge_tr, cv2.DIST_L1, 3) 
                    im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
                    im_dist = Image.fromarray(im_dist)
                    tensor_cropped = transform_A(im_dist)                    
                    dist_tensor = tensor_cropped if e == 1 else torch.cat([dist_tensor, tensor_cropped])
                    e += 1

        return im_edges, dist_tensor    


    def name(self):
        return 'FaceDataset'