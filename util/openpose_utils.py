# openpose_utils
import numpy as np
from skimage.draw import disk, line_aa, polygon, circle_perimeter_aa
from skimage import morphology
import math
import numbers
import torch

MISSING_VALUE = 0

LIMB_SEQ_25 = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], 
            [8,9], [9,10], [10,11], [11,24], [11,22], [22,23], 
            [8,12],[12,13],[13,14], [14,21], [14,19], [19,20],
            [1,0], [0,16], [16,18], [0,15],  [15,17]]

LIMB_SEQ_18 = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], 
            [8,9], [9,10], [1,11], [11,12], [12,13], 
            [1,0], [0,14], [14,16], [0,15],  [15,17]]    
                    

HAND_SEQ = [[0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],
            [0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20]]

LIMB_SEQ_HUMAN36M_17 = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],
                        [0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]

LIMB_SEQ_COCO_17 = [[0,1],[1,3],[0,2],[2,4],[5,7],[7,9],
                        [6,8],[8,10],[11,12],[5,6],[11,13],[12,14],
                        [13,15],[14,16],[5,11],[6,12]]                        

OPENPOSE_25 = { "Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,
                "LWrist":7,"MidHip":8,"RHip":9,"RKnee":10,"RAnkle":11,"LHip":12,"LKnee":13,
                "LAnkle":14,"REye":15,"LEye":16,"REar":17,"LEar":18,"LBigToe":19,"LSmallToe":20,
                "LHeel":21,"RBigToe":22,"RSmallToe":23,"RHeel":24 }

OPENPOSE_18 = { "Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,
                "LWrist":7,"RHip":8,"RKnee":9,"RAnkle":10,"LHip":11,"LKnee":12,"LAnkle":13,
                "REye":14,"LEye":15,"REar":16,"LEar":17 }                

COCO_17 =  { "Nose":0, "LEye":1, "REye":2, "LEar":3, "REar":4, "LShoulder":5, 
             "RShoulder":6, "LElbow":7, "RElbow":8, "LWrist":9, "RWrist":10, "LHip":11, 
             "RHip":12, "LKnee":13, "RKnee":14, "LAnkle":15, "RAnkle":16} 

Human36m_17 = {'Hip':0, 'RHip':1, 'RKnee':2, 'RFoot':3, 'LHip':4, 'LKnee':5, 'LFoot':6,
                'Spine':7, 'Thorax':8, 'Neck/Nose':9, 'Head':10, 'LShoulder':11,
                'LElbow':12, 'LWrist':13, 'RShoulder':14, 'RElbow':15, 'RWrist':16}                         

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 18: # CelebAMask-HQ
        cmap = np.array([[255, 0, 0],   [255, 85, 0], [255, 170, 0], [255, 255, 0], 
                         [170, 255, 0], [85, 255, 0], [0, 255, 0],
                         [0, 255, 85], [0, 255, 170], [0, 255, 255], 
                         [0, 170, 255], [0, 85, 255], [0, 0, 255],   [85, 0, 255],
                         [170, 0, 255], [255, 0, 255],[255, 0, 170], [255, 0, 85]],
                         dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        if len(gray_image.size()) != 3:
            gray_image = gray_image[0]
            
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        color_image = color_image.float()/255.0 * 2 - 1
        return color_image    



def openpose_to_map(B_coor, resize_param=None, org_size=None, sigma=6, affine=None):
    pose_joints = obtain_2d_cords(B_coor, resize_param, org_size, affine)
    if resize_param is not None:
        im_size = resize_param
    else:
        im_size = org_size  
    pose_body = pose_joints['body']
    result = obtain_map(pose_body, im_size, sigma)
    return result

def obtain_map(pose_joints, im_size, sigma=6):
    result = np.zeros([im_size[0],im_size[1],pose_joints.shape[1]], dtype='float32')
    for i in range(pose_joints.shape[1]):
        y = pose_joints[0, i]
        x = pose_joints[1, i]
        if x == MISSING_VALUE or y == MISSING_VALUE:
            continue            
        xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        result[..., i] = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    return result        

def draw_pose_from_cords(B_coor, resize_param=None, org_size=None, radius=2, affine=None, LIMB_SEQ=None):
    pose_joints = obtain_2d_cords(B_coor, resize_param, org_size, affine)
    pose_body = pose_joints['body']
    pose_body = pose_body.astype(np.int)
    if resize_param is not None:
        im_size = resize_param
    else:
        im_size = org_size

    colors = np.zeros(shape=im_size + (3, ), dtype=np.uint8)
    if LIMB_SEQ is None:
        if pose_body.shape[1] == 25:
            LIMB_SEQ = LIMB_SEQ_25
        elif pose_body.shape[1] == 18:
            LIMB_SEQ = LIMB_SEQ_18
        elif pose_body.shape[1] == 17:
            LIMB_SEQ = LIMB_SEQ_HUMAN36M_17
    colors = draw_joint(colors, pose_body, LIMB_SEQ, radius)
    return colors

def draw_joint(colors, pose_joints, joint_line_list, radius=2):
    im_size = (colors.shape[0], colors.shape[1])
    for f, t in joint_line_list:
        from_missing = pose_joints[0,f] == MISSING_VALUE or pose_joints[1,f] == MISSING_VALUE
        to_missing = pose_joints[0,t] == MISSING_VALUE or pose_joints[1,t] == MISSING_VALUE
        if from_missing or to_missing:
            continue
        yy, xx, val = line_aa(pose_joints[0,f], pose_joints[1,f], pose_joints[0,t], pose_joints[1,t])
        yy, xx = np.clip(yy, 0, im_size[0]-1), np.clip(xx, 0, im_size[1]-1)
        colors[yy, xx] = np.expand_dims(val, 1) * 255
        # mask[yy, xx] = True

    colormap = labelcolormap(pose_joints.shape[1])
    for i in range(pose_joints.shape[1]):
        if pose_joints[0,i] == MISSING_VALUE or pose_joints[1,i] == MISSING_VALUE:
            continue
        yy, xx = disk((pose_joints[0,i], pose_joints[1,i]), radius=radius, shape=im_size)
        colors[yy, xx] = colormap[i]
    return colors



def obtain_2d_cords(B_coor, resize_param=None, org_size=None, affine=None):
    pose_joints=dict()
    pose = B_coor["pose_keypoints_2d"]
    coor_x = [pose[3*i]   for i in range(int(len(pose)/3))]
    coor_y = [pose[3*i+1] for i in range(int(len(pose)/3))]
    score  = [pose[3*i+2] for i in range(int(len(pose)/3))]
    coor_body = modify_coor(coor_x, coor_y, resize_param, org_size, affine)
    pose_joints['body'] = coor_body

    return pose_joints

def modify_coor(coor_x, coor_y, resize_param=None, org_size=None, affine=None):
    out_img_size = org_size
    if resize_param is not None:
        assert org_size is not None, "org_size is required if you use resize_param" 
        for i in range(len(coor_x)):
            if coor_x[i] == MISSING_VALUE or coor_y[i] == MISSING_VALUE:
                continue
            coor_x[i] = coor_x[i]/org_size[1]*resize_param[1]
            coor_y[i] = coor_y[i]/org_size[0]*resize_param[0]
        out_img_size = resize_param  

    if affine is not None:
        center = (out_img_size[0] * 0.5 + 0.5, out_img_size[1] * 0.5 + 0.5)
        affine_matrix = get_affine_matrix(center=center, affine=affine)
        for i in range(len(coor_x)):
            if coor_x[i] == MISSING_VALUE or coor_y[i] == MISSING_VALUE:
                continue
            point_ = np.dot(affine_matrix, np.matrix([coor_x[i], coor_y[i], 1]).reshape(3,1))
            coor_y[i] = int(point_[1])
            coor_x[i] = int(point_[0])
    
    pose_joints = np.array([coor_y, coor_x])
    
    return pose_joints

def get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
    #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


    angle = math.radians(angle)
    if isinstance(shear, (tuple, list)) and len(shear) == 2:
        shear = [math.radians(s) for s in shear]
    elif isinstance(shear, numbers.Number):
        shear = math.radians(shear)
        shear = [shear, 0]
    else:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
        math.sin(angle + shear[0]) * math.sin(angle + shear[1])
    matrix = [
        math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
        -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return matrix

def get_affine_matrix(center, affine, shear=0):
    angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
    matrix_inv = get_inverse_affine_matrix(center, angle, translate, scale, shear)

    matrix_inv = np.matrix(matrix_inv).reshape(2,3)
    pad = np.matrix([0,0,1])
    matrix_inv = np.concatenate((matrix_inv, pad), 0)
    matrix = np.linalg.inv(matrix_inv)
    return matrix

def openpose25_to_coco17(pose_joints_25):
    pose_joints_17 = np.zeros((2,17)).astype(pose_joints_18.dtype)
    i = 0
    for key in COCO_17:
        pose_joints_17[:, i] = pose_joints_25[:, OPENPOSE_25[key]]
        i = i+1
    return pose_joints_17  

def openpose18_to_coco17(pose_joints_18):
    pose_joints_17 = np.zeros((2,17)).astype(pose_joints_18.dtype)
    i = 0
    for key in COCO_17:
        pose_joints_17[:, i] = pose_joints_18[:, OPENPOSE_18[key]]
        i = i+1
    return pose_joints_17  

def extract_aligned_kp(pose, pose_format):
    out_list=[]
    for key in pose_format:
        if key in ['LFoot', 'RFoot', 'LKnee', 'RKnee', 
                    'RHip', 'LHip', 'LShoulder', 'RShoulder',
                    'LAnkle', 'RAnkle']:
           out_list.append(pose_format[key])
    pose = pose[:, out_list]
    return pose

class tensor2skeleton():
    def __init__(self, image_size=(256,256), spatial_draw=False):
        super(tensor2skeleton, self).__init__()
        self.image_size=image_size
        self.use_spatial_draw=spatial_draw

    def __call__(self, tensor, kp_form='openpose_18'):
        if len(tensor.shape)==3:
            tensor=tensor[0,...]
        coors = tensor.detach().cpu().numpy()
        coors = self.denormlize(coors)
        colors_list=[]
        for i in range(coors.shape[1]):
            colors = np.zeros(shape=self.image_size + (3, ), dtype=np.uint8)
            coor = coors[:,i]


            coor = coor.reshape(-1, int(kp_form[-2:]))
            coor = coor[0:2,...]
            if kp_form == 'openpose_25':
                assert coor.shape[1] == 25
                colors = self.draw_joint_for_vis(self.image_size, coor, LIMB_SEQ_25, 2, use_spatial_draw=self.use_spatial_draw)
            elif kp_form == 'openpose_67':
                assert coor.shape[1] == 25+2*21
                colors = self.draw_joint_for_vis(self.image_size, coor[:, 0:25], LIMB_SEQ_25, 3, use_spatial_draw=self.use_spatial_draw)
                colors = self.draw_joint_for_vis(self.image_size, coor[:, 25:46], HAND_SEQ, 3, use_spatial_draw=self.use_spatial_draw)
                colors = self.draw_joint_for_vis(self.image_size, coor[:, 46:67], HAND_SEQ, 3, use_spatial_draw=self.use_spatial_draw)
            elif kp_form == 'openpose_18':
                assert coor.shape[1] == 18
                colors = self.draw_joint_for_vis(self.image_size, coor[:, 0:18], LIMB_SEQ_18, 3, use_spatial_draw=self.use_spatial_draw)
            elif kp_form == 'human36m_17':
                assert coor.shape[1] == 17
                colors = self.draw_joint_for_vis(self.image_size, coor, LIMB_SEQ_HUMAN36M_17, 3, use_spatial_draw=self.use_spatial_draw)
            elif kp_form == 'COCO_17':
                assert coor.shape[1] == 17
                colors = self.draw_joint_for_vis(self.image_size, coor, LIMB_SEQ_COCO_17, 3, use_spatial_draw=self.use_spatial_draw)
            colors_list.append(colors)
        return colors_list

    def denormlize(self, x):
        h,w = self.image_size
        x = (x+1)/2*w
        x=x.astype(np.int)
        x[x<0]=0
        x[x>255]=255
        return x

    def draw_joint_for_vis(self, image_size, pose_joints, joint_line_list, radius=2, use_spatial_draw=False):
        if use_spatial_draw:
            color = self.spatial_draw(image_size, pose_joints, joint_line_list, radius)
        else:
            color = np.zeros(shape=image_size + (3, ), dtype=np.uint8)
            color = draw_joint(color, pose_joints, joint_line_list, radius)
        return color
 
    def spatial_draw(self, image_size, pose_joints, joint_line_list, radius=2, line_color=[118,214,255], cycle_color=[66,115,177]):
        colors = np.ones(shape=image_size + (3, ), dtype=np.uint8)*255.0
        mask = np.zeros(shape=image_size, dtype=np.uint8)
        for f, t in joint_line_list:
            yy, xx, val = line_aa(pose_joints[0,f], pose_joints[1,f], pose_joints[0,t], pose_joints[1,t])
            yy[yy>image_size[0]-1]=image_size[0]-1
            xx[xx>image_size[1]-1]=image_size[1]-1
            mask[yy, xx] = 1 
        mask = morphology.dilation(mask, morphology.disk(radius=1))
        colors[mask==1] = line_color

        mask = np.zeros(shape=image_size, dtype=np.uint8)
        for i in range(pose_joints.shape[1]):
            yy, xx, val = circle_perimeter_aa(pose_joints[0,i], pose_joints[1,i], radius=radius)
            # yy, xx = disk((pose_joints[0,i], pose_joints[1,i]), radius=radius, shape=im_size)
            yy[yy>image_size[0]-1]=image_size[0]-1
            xx[xx>image_size[1]-1]=image_size[1]-1
            mask[yy, xx] = 1 
        # mask = morphology.dilation(mask, morphology.disk(radius=1))
        colors[mask==1] = cycle_color

        return colors
