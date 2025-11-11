from lib.stylizer import *

import argparse

import argparse

from pathlib import Path

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from pds.utils.trainutil import save_config
from pds.utils.sysutil import clean_gpu
from pds.utils.imageutil import resize
from pds.utils.svgutil import render, clip_curve_shape



class PDSstylizer(styleBase):
    def __init__(self,style_img,content_image,contentweight = 1,style_weight=1000):
        super().__init__(style_img,content_image)


        self.style_weight = style_weight
        self.content_weight = contentweight
        self.style_img = style_img
        from pds.pds import PDSConfig, PDS

        sd_model = 'runwayml/stable-diffusion-v1-5'
        sd_model = 'stabilityai/stable-diffusion-2-1-base'
        
        # sd_model = "qiacheng/stable-diffusion-v1-5-lcm"
        # sd_model = "SimianLuo/LCM_Dreamshaper_v7"
        pds_config_dict = {
            'sd_pretrained_model_or_path' : sd_model,
            'num_inference_steps' : 1000,
            'min_step_ratio' : 0.02,
            'max_step_ratio' : 0.5/1.0,
            # 'src_prompt' : 'a photo of a man moving',# oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed',
            # 'tgt_prompt' : 'a photo of a man moving in the style of Vincent van Gogh',# detailed high resolution, high quality, sharp',
            # 'tgt_prompt' : 'a photo of a man moving in a rainbow',
            'src_prompt' : 'A person is pushing a baby stroller, walking in front of a shop window.',
            'tgt_prompt' : 'A person is pushing a baby stroller, walking in front of a shop window in the style of Vincent van Gogh',# detailed high resolution, high quality, sharp',
            # in the impressionist style of Claude Monet
            'guidance_scale' :100.0,
            'device' : 'cuda'
        }
        pds_config = PDSConfig(**pds_config_dict)
        
        self.pds = PDS(config = pds_config)

        self.nnfm_loss_fn = nnfm_loss.NNFMLoss(device='cuda')

        c_img = self.resize_to_512(content_image.cuda())
        with torch.no_grad():
            self.src_w0 = self.pds.encode_image(c_img)

    def resize_to_512(self,img):
        # return img
        h, w = img.shape[2:]
        l = min(h, w)
        h = int(h * 512 / l)
        w = int(w * 512 / l)
        #temporal ow
        w = 904
        img_512 = F.interpolate(img, size=(h, w), mode="bilinear")
        return img_512
    
    def evalLoss(self,input_img,content_image,print_loss=False,prev_image = None):

        input_img = input_img

        
        loss_dict = self.nnfm_loss_fn(
        input_img,
        self.style_img,
        blocks=[
        2,
        ],
        loss_names=[ "content_loss"],
        contents=content_image,
        )
        loss_dict["content_loss"] *= 1e-3 * 1.0 *self.content_weight

        # loss = sum(list(loss_dict.values())) 
        
        # print(content_image.shape)
        # print(input_img.shape)

        resize_input_img = self.resize_to_512(input_img)
        w0 = self.pds.encode_image(resize_input_img)
        
        loss_args = {
            'tgt_x0' : None,
            'src_x0' : self.src_w0,
            'original_image':resize_input_img,
            'method' : "sds"
        }
        

        loss_args['tgt_x0'] = w0
        # loss_args['original_image'] = resize_input_img
        # print(resize_input_img.shape)
        loss = self.pds(**loss_args)*self.style_weight #+ loss_dict["content_loss"]

        
        if print_loss:
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
               loss.item(), loss_dict["content_loss"].item()))
            print()
        
        loss = loss #+ loss_dict["content_loss"]


        return loss
