

import torch
from lib.stylizer import *
import os
# making no connection
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
import argparse

import argparse

from pathlib import Path

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils_3DNNFM import extract_mean_colors
from lib.features import FeatureExtractor, VGGFeatures
from lib.loss_3d import nnfm_loss_3d

style_layers = [
    "layer3_0_conv2",
    "layer3_1_conv2",
    "layer3_2_conv2",
    "layer3_4_conv2",
    "layer3_5_conv2",
    # "layer4_0_conv2",
    # "layer4_1_conv2",
    # "layer4_2_conv2",
]

# Initialize for multi gpu training.
class NNFM3D(styleBase):
    """A wrapper around nn.Module to store the optimization parameters.

    Pytorch requries trainable parameters for distrubuted training. Using
    nn.Module, we are able to control the trainable parameters.
    """

    def __init__(self,style_img,content_image,contentweight = 1,style_weight=100000):
        super().__init__(style_img,content_image)
        device = 'cuda'
        clip_model, _ = clip.load("RN50", 'cuda')
        self.feature_extractor = FeatureExtractor(
        clip_model.visual.requires_grad_(False), style_layers).cuda()
        
        self.vgg_extractor = VGGFeatures(device)
        
        self.style_weight = style_weight
        self.content_weight = contentweight
        self.style_img = style_img
        with torch.no_grad():
            self.style_features = self.feature_extractor(style_img)
            
            self.mean_colors =  extract_mean_colors(style_img).cuda()
            
    def evalLoss(self,input_img,content_image,print_loss=False,prev_image = None):
        123
        target_image = input_img
        target_style_features = self.feature_extractor(target_image)

        style_loss = self.style_weight * nnfm_loss_3d(self.style_features, target_style_features)
        with torch.no_grad():
            src_content_features = self.vgg_extractor(content_image)
        target_content_features = self.vgg_extractor(target_image)

        content_loss = 0.0
        for (tc, sc) in zip(target_content_features, src_content_features):
            content_loss = content_loss + self.content_weight * F.mse_loss(tc, sc)
        
        # Compute the color loss
        color_image = target_image.permute(0, 2, 3, 1)
        color_image = color_image.view(-1, 3)
        
        num_iterations = 1.0 
        
        weight = 1.0
        color_losses = torch.cdist(color_image, self.mean_colors).min(dim=1)[0]
        mean_loss = weight * color_losses.mean()
        # var_loss = weight * color_losses.var()
        color_loss = mean_loss
        total_loss = style_loss + content_loss  + color_loss
        if print_loss:
            print(total_loss.item())
        
        return total_loss


 
