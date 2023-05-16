import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms
import torchvision.utils as tv_utils
import kornia

def rgb2lab(img):
    '''
        convert image to Lab color space

        Args:
            img(torch.Tensor (B, C, H, W)): RGB image, range (0,1)
        
        Returns:
            torch.Tensor (B, C, H, W): Lab image

    '''

    # get batchsize from image
    batchsize, C, H, W = img.size(0), img.size(1), img.size(2), img.size(3)
    # convert rgb to lab
    lab_batch = kornia.color.rgb_to_lab(img)
    
    return lab_batch

def lab_to_rgb(lab_batch):
    """
    Convert a batch of Lab images to RGB.

    Args:
        lab_batch (torch.Tensor): A tensor of shape (batch_size, 3, height, width) representing
            a batch of Lab images. The L channel is assumed to be in the range of \([0, 100]\). a and b channels are in the range of \([-128, 127]\)

    Returns:
        A tensor of shape (batch_size, C, height, width) representing the converted RGB images. The output RGB image are in the range of \([0, 1]\).
    """
    
    # convert lab to rgb
    rgb = kornia.color.lab_to_rgb(lab_batch)
    
    return rgb

def test():
    batchsize = 1

    #load dataset
    path = r"decompose\dataset"
    dataset = datasets.ImageFolder(path, transform = torchvision.transforms.ToTensor())
    
    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

    for batch in dataloader:
        lab = rgb2lab(batch[0])
        lab_show = torch.squeeze(lab,0)
        tv_utils.save_image(lab_show, "lab_show.png", normalize=True)
        rgb = lab_to_rgb(lab)
        rgb_show = torch.squeeze(rgb,0)
        tv_utils.save_image(rgb_show, "rgb_show.png", normalize=True)
        # tv_utils.save_image(batch[0],"onriginal.png", normalize=True)

test()