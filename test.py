from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch

if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    # model.eval()
    # create a visualizer
    # visualizer = visualizer.Visualizer(opt)

    # for i, data in enumerate(islice(dataset, opt.how_many)):
    #     model.set_input(data)
    #     model.test()

    # for i, data in enumerate(islice(dataset, opt.how_many)):
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()


