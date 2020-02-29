from block_extractor import BlockExtractor
import torch
from PIL import Image
import torchvision.transforms as transforms
import imageio
import numpy as np


def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0,:3,:,:].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

transform_list=[]
label_path_1 = '/home/yurui/dataset/celeba/celeba/190688.jpg'
label_path_2 = '/home/yurui/dataset/celeba/celeba/190687.jpg'
if __name__ == "__main__":
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # forward check
    kernel_size = 3
    extractor = BlockExtractor(kernel_size)

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    tran = transforms.Compose(transform_list)

    label_1 = Image.open(label_path_1)
    label_2 = Image.open(label_path_2)

    tensor_1 = tran(label_1).unsqueeze(0)
    tensor_2 = tran(label_2).unsqueeze(0)
    image = torch.cat((tensor_1, tensor_2), 0)
    source = image.cuda()
    flow = torch.ones(source.shape).cuda()*0
    flow = flow[:,:2,:,:] 

    out = extractor(source, flow)

    image = tensor2im(out)
    save_image(image, 'test.png')
    image = tensor2im(source)
    save_image(image, 'source.png')
    # print(torch.sum(torch.abs(out[0,0,3:6,3:6]-source[0,0,0:3,0:3])))
    # image = tensor2im(out-source)
    # save_image(image, 'mine.png')    





    # source = torch.rand(8,3,64,64).cuda()
    # flow = torch.rand(8,2,64,64).cuda()*0
    # source.requires_grad=True
    # flow.requires_grad=True
    # out = markovattn(source, flow)
    # print(torch.max(torch.abs(out-source)))
    # print(torch.min(torch.abs(out-source)))
    # image = tensor2im(out-source)
    # save_image(image, 'test.png')

    # backward check
    source = torch.rand(4,6,14,10).double().cuda()
    flow = torch.rand(4,2,14,10).double().cuda()*1.8
    source.requires_grad=True
    flow.requires_grad=True
    print(torch.autograd.gradcheck(extractor, (source, flow)) )



