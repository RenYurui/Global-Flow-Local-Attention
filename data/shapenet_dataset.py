import torchvision.transforms as transforms
from PIL import Image
import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np
import torch
import h5py

class ShapeNetDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--ang_skip', type=int, default=20 )
        parser.add_argument('--label_nc_h', type=int, default=18 )
        parser.add_argument('--label_nc_v', type=int, default=3 )
        parser.add_argument('--sub_dataset_model', type=str, default='car')

        parser.set_defaults(load_size=256)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(structure_nc=18+3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_id_file, self.hdf5_file, self.image_name_file = self.get_paths(opt)

        self.image_ids = np.genfromtxt(self.image_id_file, dtype=np.str)
        self.dataset_size = len(self.image_ids)
        transform_list=[]
        transform_list.append(transforms.Resize(size=(opt.load_size, opt.load_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 
        self.hdf5_data = None
        # self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        self.angle_list = range(0, 360, opt.ang_skip)
        if not self.opt.isTrain:
            self.image_names = np.genfromtxt(self.image_name_file, dtype=np.str)
            np.random.seed(5)


    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'val' else opt.phase
        image_id_file  = os.path.join(root, 'id_%s_%s.txt' %(opt.sub_dataset_model, phase))
        image_name_file  = os.path.join(root, 'name_%s_%s.txt' %(opt.sub_dataset_model, phase))
        hdf5_file = os.path.join(root, 'data_%s.hdf5'% opt.sub_dataset_model)
        return image_id_file, hdf5_file, image_name_file     

    def __getitem__(self, index):
        if self.hdf5_data is None:
            # follow the solution at 
            # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
            # to sovle multi-thread read of HDF5 file 
            self.hdf5_data = h5py.File(self.hdf5_file, 'r')

        if self.opt.isTrain:
            source_id = self.image_ids[index]
            source_id = source_id.decode("utf-8") if isinstance(source_id, bytes) else source_id
            target_id = self.get_random_target_id(source_id)


            P1_img = self.hdf5_data[source_id]['image'][()]
            P2_img = self.hdf5_data[target_id]['image'][()]

            P1_img = Image.fromarray(np.uint8(P1_img))
            P2_img = Image.fromarray(np.uint8(P2_img))

            P1 = self.trans(P1_img)
            P2 = self.trans(P2_img)

            BP1 = torch.tensor(self.hdf5_data[source_id]['pose'][()]).view(-1, 1, 1)
            BP2 = torch.tensor(self.hdf5_data[target_id]['pose'][()]).view(-1, 1, 1)

        else:
            source_names = self.image_names[index]
            source_names = source_names.decode("utf-8") if isinstance(source_names, bytes) else source_names

            random_h_angle = str(int(self.angle_list[index%len(self.angle_list)]/10))
            random_v_angle = str(0)            
            source_id = source_names+ '_' + random_h_angle + '_' + random_v_angle

            P1_img = self.hdf5_data[source_id]['image'][()]
            P1_img = Image.fromarray(np.uint8(P1_img))
            P1 = self.trans(P1_img)
            BP1 = torch.tensor(self.hdf5_data[source_id]['pose'][()]).view(-1, 1, 1)

            P2 = []
            BP2 = []
            target_id=[]

            for ang in self.angle_list:
                t_b = torch.LongTensor([int(ang/10),int(random_v_angle)]).view(-1, 1, 1)
                t_id=source_names+ '_' + str(int(ang/10)) + '_' + random_v_angle
                t_img = self.hdf5_data[t_id]['image'][()]
                t_img = Image.fromarray(np.uint8(t_img))
                t_img = self.trans(t_img)

                BP2.append(t_b)
                target_id.append(t_id)
                P2.append(t_img)
        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 
                'P1_path': source_id, 'P2_path': target_id}



    def get_random_target_id(self, source_id):
        target_angle = int(np.random.choice(self.angle_list)/10)
        id_base = source_id.split('_')[0]
        h = source_id.split('_')[-1]
        target_id = '_'.join([id_base, str(target_angle), str(h)])
        return target_id

    def __len__(self):
        if self.opt.isTrain:
            return len(self.image_ids)
        else:
            return len(self.image_names)

