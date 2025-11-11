import torch
import os
import icecream as ic
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import imageio

import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)

imsize = 512


import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as visionF
import torchvision.transforms as T

def plotHelper(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = visionF.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
import cv2

def image_loader(image_name,resize = False,resoultion = 400,align_to_dual = False,resize_arf = False,half_resize=True,content_long_side = 768):
    if resize and not resize_arf:
        loader = transforms.Compose([
        transforms.Resize(resoultion),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    else:
        loader = transforms.Compose([
       #  transforms.Resize(resoultion),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor 
    
    image = Image.open(image_name)
    style_img = loader(image).unsqueeze(0)
    print(style_img.shape)
    style_h = style_img.shape[-2]
    style_w = style_img.shape[-1]
    if resize_arf:
        content_long_side = 768
        # print((int(content_long_side / style_h * style_w), content_long_side))
        if style_h > style_w:
            style_img = cv2.resize(
                style_img.permute(0,2,3,1).cpu().numpy().squeeze(0),
                (int(content_long_side / style_h * style_w), content_long_side),
                interpolation=cv2.INTER_AREA,
            )
        else:
            style_img = cv2.resize(
                style_img.permute(0,2,3,1).cpu().numpy().squeeze(0),
                (content_long_side, int(content_long_side / style_w * style_h)),
                interpolation=cv2.INTER_AREA,
            )
        if half_resize:
            style_img = cv2.resize(
                style_img,
                (style_img.shape[1] // 2, style_img.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
        style_img = torch.tensor(style_img).unsqueeze(0).permute(0,3,1,2)
    image = style_img
    
    # fake batch dimension required to fit network's input dimensions
    
    if align_to_dual:
        H = image.shape[-1]
        W = image.shape[-2]

        H = (H+1) // 2 * 2
        W = (W+1) // 2 * 2
        
        _resize = transforms.Resize((W, H), interpolation=transforms.InterpolationMode.BILINEAR)
        image = _resize(image)
    
    return image.to(device, torch.float)


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def save_Tensor_image(tesnor,name):
    rgb8 = to8b(tesnor.squeeze(0).permute(1, 2, 0).cpu().numpy() )
    imageio.imwrite(name, rgb8)
    
    
