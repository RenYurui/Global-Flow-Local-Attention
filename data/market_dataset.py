import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch
import torchvision.transforms.functional as F

class MarketDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=(128, 64))
            parser.set_defaults(angle=(-5, 5))
            parser.set_defaults(shift=(-5, 5))
            parser.set_defaults(scale=(0.95, 1.05))

        else:
            parser.set_defaults(load_size=(128, 64))
            parser.set_defaults(angle=False)
            parser.set_defaults(shift=False)
            parser.set_defaults(scale=False)            
        parser.set_defaults(old_size=(128, 64))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=128)



        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'market-pairs-%s.csv' % phase)
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'market-annotation-%s.csv' % phase)

        return image_dir, bonesLst, name_pairs        

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs   


    def name(self):
        return "MarketDataset"  


