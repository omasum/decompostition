import numpy as np
from guided_filter_pytorch.guided_filter import GuidedFilter
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.utils as tv_utils
import kornia
import os
from torchvision import datasets
import torchvision.transforms

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


def guided_filter(p, radius, epsilon=0.1, I=None):
    '''
        guide-filter image , when there is no guide image I, I = p 

        parameters:
        p: original image, torch.Tensor (B, C, H, W)
        radius: r parameter, int
        epsilon: eps parameter, float
        I: guide image, torch.Tensor (B, C, H, W)

        return:
        filtered image, torch.Tensor (B, C, H, W)
    '''
    if radius < 1:
        print("radius should be equal to or larger than 1.")
        return p
    if I is None:
        I = p
    q = GuidedFilter(radius, epsilon)(I, p)

    return q

def gf_decompose(im, mask=None, log_epsilon=0.0001, filter_epsilon=0.01, logspace=True):
    '''
        Args:
        im: torch.Tensor [B, C, H, W], rgb image
        mask: None
        log_epsilon: float, default = 0.0001
        filter_epsilon: float, default = 0.01
        logspace: True or False

        Return:
        dec: dict, 
        {
        "size": image shape, 
        "depth": decomposition subbands amount 
        "n_color": color channel amount (3)
        "log_epsilon": log_epsilon 
        "filter_epsilon": parameter of guided filter
        "L": a list of N+1 dictionaries, D1...DN: N dictionary subbands decomposition images( 4 elements ), residual: last scale guided-filter result
        "a": a channel of lab image
        "b": b channel of lab image
        "L_origin": L channel of lab image (origin)
        }
    '''

    C = im.shape[1]
    if C == 2 or C > 3:
        print("The number of color channel must be either 1 or 3.")
        return None
    elif C == 3:
        lab = rgb2lab(im)
        lab_L = torch.split(lab, 1, dim = 1)[0]
        lab_a = torch.split(lab, 1, dim = 1)[1]
        lab_b = torch.split(lab, 1, dim = 1)[2]
        dec = gf_decompose(lab_L, mask, log_epsilon, filter_epsilon, logspace)
        dec.update({'a': lab_a, 'b': lab_b})
        dec.update({'L_origin': lab_L})
        dec['n_color'] = 3
        return dec

    dec = gf_decompose_scale(im, None, log_epsilon, filter_epsilon, logspace)
    dec = gf_decompose_parts(dec, mask)
    return dec

def gf_decompose_scale(im, depth=None, log_epsilon=0.0001, filter_epsilon=0.01, logspace=True):

    '''

        single-channel image "im" decomposition, using mode of band-sifting, subband is achieved by making difference of two guide-filter of double-scale

        Args:
            im: torch.tensor(B, 1, H ,W), single-channel image C (L channel), range (0, 100)
            depth: torch.tensor, dtype = torch.int32, decomposition depth, the number of subbands
            log_epsilon: float32, If TRUE (default), image processing is done in the log space. If FALSE, computation is performed without log transformation.
            filter_epsilon: float32, guided filter parameter
            logspace: True or False, If TRUE (default), image processing is done in the log space. If FALSE, computation is performed without log transformation.

        return:
            dec: dict, 
            {
                "size": image shape,
                "depth": decomposition subbands amount
                "n_color": color channel amount
                "log_epsilon": log_epsilon
                "filter_epsilon": parameter of guided filter
                "L": a dictionary of subband images, D1...DN: N subbands decomposition images, residual: last scale guided-filter result
            }

    '''

    # when image H = W, min length of side = H
    H = torch.tensor(im.shape[2]) 
    if depth is None:
        depth = (torch.floor(torch.log2(H)) - 1).int()

    # L
    if logspace:
        L = torch.log(im + log_epsilon)
    else:
        L = im

    # decomposition of L
    if depth == 0:
        N = 0
        D = {"residual": L}
    else:
        N = depth
        L_k_minus_1 = guided_filter(L, 2 ** 1, filter_epsilon)
        D_k = L - L_k_minus_1
        D = {"D1": D_k}
        if N > 1:
            for k in range(1, N):
                L_k = guided_filter(L_k_minus_1, 2 ** k, filter_epsilon)
                D_k = L_k_minus_1 - L_k
                D[f"D{k:02d}"] = D_k
                if k == N-1:
                    D["residual"] = L_k
                else:
                    L_k_minus_1 = L_k
        elif N == 1:
            D["residual"] = L_k_minus_1
    
    dec = {"size": im.shape, # torch.Size
           "depth": N, # torch.Tensor
           "n_color": 1,
           "log_epsilon": log_epsilon, # torch.float32
           "filter_epsilon": filter_epsilon, # torch.float32
           "L": D} # dictionary
    return dec

def cubic_spline(x, low=0, high=1):
    if low == high:
        print("low and high must be different!")
    elif low > high:
        return 1 - cubic_spline(x, high, low)
    x2 = x.clone()
    t = x[(x > low) & (x < high)]
    t = (t - low) / (high - low)
    x2[(x > low) & (x < high)] = t ** 2 * (3 - 2 * t)
    x2[x <= low] = 0
    x2[x >= high] = 1
    return x2

def gf_decompose_parts(dec, mask=None):
    '''
        decompose dec to sign and amplitude criteria( positive, negtive, high, low), finally there depth*4 subbands

        Args:
            dec: dict, information after scale criteria

        Return:
            dec: dict, 
            {
                "size": image shape
                "depth": decomposition subbands amount
                "n_color": color channel amount
                "log_epsilon": log_epsilon
                "filter_epsilon": parameter of guided filter
                "L": a list of N+1 dictionaries, D1...DN: N dictionary subbands decomposition images( 4 elements ), residual: last scale guided-filter result
            } 
            
    '''
    L = dec["L"]
    residual = L["residual"]
    del L["residual"]
    L = [im for im in L.values()]
    results = []
    blur_range = 0.2
    range_lo = 1 - blur_range
    range_hi = 1 + blur_range
    for im in L:
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                sigma = mask[mask > 0.5].std()
            else:
                sigma = im.std()
        else:
            sigma = im.std()
     
        hi = im * (cubic_spline(im, range_lo * sigma, range_hi * sigma) +
                   cubic_spline(im, -range_lo * sigma, -range_hi * sigma))
        lo = im * torch.min(cubic_spline(im, -range_hi * sigma, -range_lo * sigma),
                            cubic_spline(im, range_hi * sigma, range_lo * sigma))
        hip = torch.max(hi, torch.zeros_like(hi))
        hin = torch.min(hi, torch.zeros_like(hi))
        lop = torch.max(lo, torch.zeros_like(lo))
        lon = torch.min(lo, torch.zeros_like(lo))
        results.append({
            "highamp_posi": hip,
            "highamp_nega": hin,
            "lowamp_posi": lop,
            "lowamp_nega": lon
        })
    results.append({"residual": residual})
    dec["L"] = results
    return dec

def Lab2sRGB(img):
    '''
        convert Lab image to RGB color space

        Args:
            img(torch.Tensor( B, C, H, W)): Lab image, The L channel is assumed to be in the range of \([0, 100]\). a and b channels are in the range of \([-128, 127]\).
        
        Returns:
            rgb_batch: torch.Tensor(B, C, H, W): rgb image, range(0,1)
    '''

    rgb_batch = kornia.color.lab_to_rgb(img)
    return rgb_batch

def gf_reconstruct(dec, maps, scales, ind, include_residual=True, logspace=True):

    '''
        Reconstruct the original image from decomposed data

        Args:
            dec: decomposed data
            scales: which spatial scales to use for reconstruction, BS or initial deconposition(None)
            ind: a numeric vector
            include_residual: either TRUE (default) or FALSE
            logspace: If TRUE (default), image processing is done in the log space. If FALSE, computation is performed without log transformation.

        Return:
            rgb images: tensor[b,c,h,w], range (0,1)
    '''

    index = ["highamp_posi", "highamp_nega", "lowamp_posi", "lowamp_nega"]
    
    recon = torch.zeros(dec['size'])

    
    if scales == "BS":
        for key, value in maps.items():
            recon = recon + value

    if ind is None:
        ind = range(1, 5)

    if scales is None:
        scale = range(1, dec['depth'] + 1)
        for i in scale:
            if isinstance(dec['L'][i - 1], torch.Tensor):
                # scale-only decomposition
                recon = recon + dec['L'][i - 1]
            else:
                # scale and parts decomposition
                for j in ind:
                    idx = index[j-1]
                    recon = recon + dec['L'][i - 1][idx]

    if include_residual:
        recon = recon + dec['L'][dec['depth']]['residual']
    if logspace:
        recon = torch.exp(recon) - dec['log_epsilon']

    if dec['n_color'] == 3:
        # recon = torch.cat([recon * 100, dec['a'], dec['b']], dim=1)
        recon = torch.cat([recon, dec['a'], dec['b']], dim=1)
        recon = Lab2sRGB(recon)

    return recon


def get_BS_energy(im, mask=None, logspace=True):

    '''
        decompose using band-sift criteria, 8 subbands

        Args:
        im: torch.Tensor [B, C, H, W], rgb image

        Return:
        maps: dictionary storing 8 subbands
        dec: dictionary storing all decomposition bands and information
    '''

    # Image decomposition by the Band-Sift algorithm
    log_epsilon=0.0001
    filter_epsilon=0.01
    dec = gf_decompose(im, mask, log_epsilon, filter_epsilon, logspace)

    # BS feature maps
    maps = {
        "HHP": dec["L"][0]["highamp_posi"],
        "HHN": dec["L"][0]["highamp_nega"],
        "HLP": dec["L"][0]["lowamp_posi"],
        "HLN": dec["L"][0]["lowamp_nega"],
        "LHP": dec["L"][dec["depth"]-1]["highamp_posi"],
        "LHN": dec["L"][dec["depth"]-1]["highamp_nega"],
        "LLP": dec["L"][dec["depth"]-1]["lowamp_posi"],
        "LLN": dec["L"][dec["depth"]-1]["lowamp_nega"],
    }
    for i in range(1, dec["depth"]-1):
        if i <= dec["depth"] // 2:
            maps["HHP"] += dec["L"][i]["highamp_posi"]
            maps["HHN"] += dec["L"][i]["highamp_nega"]
            maps["HLP"] += dec["L"][i]["lowamp_posi"]
            maps["HLN"] += dec["L"][i]["lowamp_nega"]
        else:
            maps["LHP"] += dec["L"][i]["highamp_posi"]
            maps["LHN"] += dec["L"][i]["highamp_nega"]
            maps["LLP"] += dec["L"][i]["lowamp_posi"]
            maps["LLN"] += dec["L"][i]["lowamp_nega"]
    return maps, dec

def LEdit(img, maps, weight):
    '''edit decomposition bands in maps with weights

    Args:
        img: (tensor.Float[batchsize, 3, h, w]) original image
        maps: (dictionary value: tensor.Float[batchsize, 1, h, w]) maps after BS decomposition
        weight: (tensor.Float[batchsize, 8])

    Returns:
        new_maps: (dictionary value: tensor.Float[batchsize, 1, h, w])
    '''
    h,w = img.shape[2:4]
    weights = weight.t().unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, 1, h, w)
    i = 0
    new_maps = {}
    for key, value in maps.items():
        new_maps[key] = value*weights[i]
        i = i+1 # i==8 stop
    return new_maps

def Show_subbands(img, Logspace=False):
    '''
        show decomposion of img
        Args:
            img: torch.Tensor(B, C, H, W)
    '''

    maps, dec = get_BS_energy(img, logspace=Logspace)

    rebuild = gf_reconstruct(dec, maps, scales=None, ind=None, logspace=Logspace)
    rebuild_map = gf_reconstruct(dec, maps, scales="BS", ind=None, logspace=Logspace)
    weight0 = torch.zeros(img.shape[0], 8)
    weight1 = torch.ones(img.shape[0], 8)
    new_map0 = LEdit(img, maps, weight0)
    new_map1 = LEdit(img, maps, weight1)
    rebuild_map0 = gf_reconstruct(dec, new_map0, scales="BS", ind=None, logspace=Logspace)
    rebuild_map1 = gf_reconstruct(dec, new_map1, scales="BS", ind=None, logspace=Logspace)
    # for i in range(0, rebuild.shape[0]):
    #     img_rebuild = rebuild[i] # [C, H, W]
    #     img_rebuild = torch.transpose(img_rebuild, (1,2,0))
    #     cv2.imwrite(os.path.join(r'decompose\results', str(i+1)+ "_rebuild.png"), img_rebuild)

    tv_utils.save_image(rebuild, os.path.join(r'decompose\results', 'rebuild_image.png'), normalize=True)
    tv_utils.save_image(rebuild_map, os.path.join(r'decompose\results', 'rebuild_map.png'), normalize=True)
    tv_utils.save_image(rebuild_map0, os.path.join(r'decompose\results', 'rebuild_map0.png'), normalize=True)
    tv_utils.save_image(rebuild_map1, os.path.join(r'decompose\results', 'rebuild_map1.png'), normalize=True)
    L_channel = dec['L_origin']
    a_channel = dec["a"] # torch.Tensor(B, 1, H, W)
    b_channel = dec["b"] # torch.Tensor(B, 1, H, W)
    log_epsilon = 0.0001

    tc = torch.cat([L_channel, a_channel, b_channel], dim = 1)
    rgb_img = Lab2sRGB(tc) # torch.Tensor(B, 3, H, W)
    # for i in range(0, rgb_img.shape[0]):
    #     img_original = rgb_img[i]
    #     cv2.imwrite(os.path.join(r'decompose\results', str(i+1)+ "_original.png"), img_original)

    tv_utils.save_image(rgb_img, os.path.join(r'decompose\results', 'original_image.png'), normalize=True)

    contain = []

    for item in maps.items(): # key: string, value: torch.Tensor(B, 1, H, W)
        L = item[1] # torch.Tensor(B, 1, H, W)
        # L = torch.exp(L) - log_epsilon
        L_show = L.repeat(1,3,1,1) # torch.Tensor(B, 3, H, W)                                                                                                                                                       
        # rgb_img = Lab2sRGB(L_show)
        tag = item[0] #string

        for i in range(0, L_show.shape[0]):
            img_bands  = L_show[i] # torch.Tensor(3, H, W)
            # original = rgb_img[i]  # torch.Tensor(3, H, W)
            # img = torch.cat([original, img_bands],dim=0) # torch.Tensor(3*9, H, W)
            # s_img = torch.split(img_bands, 3, dim=0)
            # n_img = torch.stack(s_img, dim=0) #torch.Tensor(9, 3, H, W)
            tv_utils.save_image(img_bands, os.path.join(r'decompose\results', str(i+1)+ "_" + tag + '.png'), normalize=True)

def test():
    batchsize = 1

    #load dataset
    path = r"decompose\dataset"
    dataset = datasets.ImageFolder(path, transform = torchvision.transforms.ToTensor())
    
    #create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)

    for batch in dataloader:
        Show_subbands(batch[0], Logspace=True)

test()