import os, ntpath
import numpy as np
import torch
from collections import OrderedDict
from util import util, pose_utils
from model.networks import base_function
import matplotlib.pyplot as plt


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.value_names = []
        self.image_paths = []
        self.optimizers = []
        self.schedulers = []

    def name(self):
        return 'BaseModel'

    @staticmethod
    def modify_options(parser, is_train):
        """Add new options and rewrite default values for existing options"""
        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps"""
        pass

    def eval(self):
        pass

    def setup(self, opt):
        """Load networks, create schedulers"""
        if self.isTrain:
            self.schedulers = [base_function.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            print('model resumed from %s iteration'%opt.which_iter)
            self.load_networks(opt.which_iter)

    def set_model_to_eval_mode(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def set_model_to_train_mode(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()                  

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rate"""
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate=%.7f' % lr)

    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def get_current_eval_results(self):
        """Return training loss"""
        eval_ret = OrderedDict()
        for name in self.eval_metric_name:
            if isinstance(name, str):
                eval_ret[name] = getattr(self, 'eval_' + name).item()
        return eval_ret

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    # visual multi-scale ouputs
                    for i in range(len(value)):
                        visual_ret[name + str(i)] = self.convert2im(value[i], name)
                    # visual_ret[name] = util.tensor2im(value[-1].data)       
                else:
                    visual_ret[name] =self.convert2im(value, name)         
        return visual_ret        

    def convert2im(self, value, name):
        if 'label' in name:
            convert = getattr(self, 'label2color')
            value = convert(value)

        if 'flow' in name: # flow_field
            convert = getattr(self, 'flow2color')
            value = convert(value)

        if value.size(1) == 18: # bone_map
            value = np.transpose(value[0].detach().cpu().numpy(),(1,2,0))
            value = pose_utils.draw_pose_from_map(value)[0]
            result = value

        elif value.size(1) == 21: # bone_map + color image
            value = np.transpose(value[0,-3:,...].detach().cpu().numpy(),(1,2,0))
            # value = pose_utils.draw_pose_from_map(value)[0]
            result = value.astype(np.uint8)            

        elif value.size(1) == 16 and 'flow' not in name: # face map
            value = np.transpose(value[0,0,...].unsqueeze(0).detach().cpu().numpy(),(1,2,0))
            result = (value*255).astype(np.uint8)

        else:
            result = util.tensor2im(value.data)
        return result

    def get_current_dis(self):
        """Return the distribution of encoder features"""
        dis_ret = OrderedDict()
        value = getattr(self, 'distribution')
        for i in range(1):
            for j, name in enumerate(self.value_names):
                if isinstance(name, str):
                    dis_ret[name+str(i)] =util.tensor2array(value[i][j].data)

        return dis_ret

    # save model
    def save_networks(self, which_epoch):
        """Save all the networks to the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    # load models
    def load_networks(self, which_epoch):
        """Load all the networks from the disk"""
        for name in self.model_names:
            if isinstance(name, str):
                filename = '%s_net_%s.pth' % (which_epoch, name)
                path = os.path.join(self.save_dir, filename)
                net = getattr(self, 'net_' + name)
                try:
                    net.load_state_dict(torch.load(path))
                    print('load %s from %s' % (name, filename))
                except FileNotFoundError:
                    print('do not find checkpoint for network %s'%name)
                    continue
                except:
                    pretrained_dict = torch.load(path)
                    model_dict = net.state_dict()
                    try:
                        pretrained_dict_ = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                        if len(pretrained_dict_)==0:
                            pretrained_dict_ = {k.replace('module.',''):v for k,v in pretrained_dict.items() if k.replace('module.','') in model_dict}
                        if len(pretrained_dict_)==0:
                            pretrained_dict_ = {('module.'+k):v for k,v in pretrained_dict.items() if 'module.'+k in model_dict}

                        pretrained_dict = pretrained_dict_
                        net.load_state_dict(pretrained_dict)
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                    except:
                        print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                # not_initialized.add(k)
                                not_initialized.add(k.split('.')[0])
                        print(sorted(not_initialized))
                        net.load_state_dict(model_dict)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()
                if not self.isTrain:
                    net.eval()
                self.opt.iter_count = util.get_iteration(self.save_dir, filename, name)   


    # def save_results(self, save_data, score=None, data_name='none'):
    #     """Save the training or testing results to disk"""
    #     img_paths = self.get_image_paths()

    #     for i in range(save_data.size(0)):
    #         print('process image ...... %s' % img_paths[i])
    #         short_path = ntpath.basename(img_paths[i])  # get image path
    #         name = os.path.splitext(short_path)[0]
    #         if type(score) == type(None):
    #             img_name = '%s_%s.png' % (name, data_name)
    #         else:
    #             # d_score = score[i].mean()
    #             # img_name = '%s_%s_%s.png' % (name, data_name, str(round(d_score.item(), 3)))
    #             img_name = '%s_%s_%s.png' % (name, data_name, str(score))
    #         # save predicted image with discriminator score
    #         util.mkdir(self.opt.results_dir)
    #         img_path = os.path.join(self.opt.results_dir, img_name)
    #         # print(os.path.dirname(img_path))
    #         # util.mkdir(os.path.dirname(img_path))
    #         img_numpy = util.tensor2im(save_data[i].data)
    #         util.save_image(img_numpy, img_path)



    def save_results(self, save_data, data_name='none', data_ext='jpg'):
        """Save the training or testing results to disk"""
        img_paths = self.get_image_paths()

        for i in range(save_data.size(0)):
            print('process image ...... %s' % img_paths[i])
            short_path = ntpath.basename(img_paths[i])  # get image path
            name = os.path.splitext(short_path)[0]
            img_name = '%s_%s.%s' % (name, data_name, data_ext)

            util.mkdir(self.opt.results_dir)
            img_path = os.path.join(self.opt.results_dir, img_name)
            img_numpy = util.tensor2im(save_data[i].data)
            util.save_image(img_numpy, img_path)            

    def save_feature_map(self, feature_map, save_path, name, add):
        if feature_map.dim() == 4:
            feature_map = feature_map[0]
        name = name[0]
        feature_map = feature_map.detach().cpu().numpy()
        for i in range(feature_map.shape[0]):
            img = feature_map[i,:,:]
            img = self.motify_outlier(img)
            name_ = os.path.join(save_path, os.path.splitext(os.path.basename(name))[0], add+'modify'+str(i)+'.png')
            util.mkdir(os.path.dirname(name_))
            plt.imsave(name_, img, cmap='viridis')


    def motify_outlier(self, points, thresh=3.5):

        median = np.median(points)
        diff = np.abs(points - median)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        outliers = points[modified_z_score > thresh]
        outliers[outliers>median] = thresh*med_abs_deviation/0.6745 + median
        outliers[outliers<median] = -1*(thresh*med_abs_deviation/0.6745 + median)
        points[modified_z_score > thresh]=outliers

        return points

        