from local_attn_reshape import LocalAttnReshape
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
if __name__ == "__main__":
    # forward check
    kernel_size = 3
    extractor = LocalAttnReshape(kernel_size)
    inputs = torch.tensor(range(9))
    print(inputs)
    inputs = inputs.view(1, -1, 1, 1).repeat(2,1,10,10).float()
    print(inputs.size())




    source = inputs.cuda()

    out = extractor(source)

    image = tensor2im(out/9.0)
    save_image(image, 'test.png')
    print(out[0,0,:3,:3])
    print(out.size())
    # image = tensor2im(source)
    # save_image(image, 'source.png')
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
    source = torch.rand(4,9,14,10).double().cuda()
    # flow = torch.rand(4,2,14,10).double().cuda()*1.8
    source.requires_grad=True
    # flow.requires_grad=True
    print(torch.autograd.gradcheck(extractor, (source)) )



