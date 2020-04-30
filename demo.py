from options.val_options import ValOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch

if __name__=='__main__':
    opt = ValOptions().parse()
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('val images = %d' % dataset_size)
    model = create_model(opt)

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()


