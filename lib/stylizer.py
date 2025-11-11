

from VGGUtils import *
import torch
import torch.nn.functional as F
from lib import nnfm_loss


import numpy as np

import torch
import torch.nn
from torchvision import transforms, models

import os
# making no connection
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

import clip

from PIL import Image 

from torchvision.transforms.functional import adjust_contrast

from lib.ClipHelper import *
from lib.template import imagenet_templates
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class styleBase():
    def __init__(self,style_img,content_image):
        self.ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        self.enable_ssim = False
        self.enable_Random_Perspective = False
        self.Radom_perspective = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=0.8,distortion_scale=0.2),
        ])
        pass

    def evalLoss(self,input_img,content_image,print_loss=False):
        raise ValueError("should not reached")

    def DirectForwardLoss(self,imageComposeMethod,content_image,print_loss=False,prev_image = None):
        input_img = imageComposeMethod() 
        prev_loss = 0.0   
        if prev_image != None and self.enable_ssim:
            # print(input_img.shape)
            # print(prev_image.shape)
            prev_loss = 1.0 - self.ssim_loss(input_img,prev_image)
            if print_loss:
                print('ssim_loss')
                print(prev_loss.item())
        loss = self.evalLoss(input_img,content_image,print_loss)
        return loss+prev_loss * 30000.0

    # def IndirectForwardLoss(self,imageComposeMethod,content_image,print_loss=False,prev_image = None):
    #     with torch.no_grad():
    #         input_img = imageComposeMethod()
    #     # input_img.re 
    #     input_img.requires_grad_(True)  
        
    #     loss = self.evalLoss(input_img,content_image,print_loss)
    #     loss.backward()
    #     RGBLoss = input_img.grad.detach().clone()
    #     del input_img
    #     input_img_wg = imageComposeMethod()
        
    #     input_img_wg.backward(gradient=RGBLoss) 
        
    #     # loss = torch.sum(input_img_wg * RGBLoss)
        
    #     return torch.tensor(0.0)

    def IndirectForwardLoss(self,imageComposeMethod,content_image,print_loss=False,prev_image = None,return_img = False):
        with torch.no_grad():
            input_img = imageComposeMethod()
        # detach to guarantee leaf tensor for gradient capture
        input_img = input_img.detach()
        input_img.requires_grad_(True)
        input_img.retain_grad()
        if self.enable_Random_Perspective:
            total_loss = 0.0
            for i in range(5):
                input_img_rp = self.Radom_perspective(input_img)
                
                loss = self.evalLoss(input_img_rp,content_image,print_loss and i ==4)
                total_loss = total_loss + loss
        else:
            total_loss = self.evalLoss(input_img,content_image,print_loss)
        total_loss.backward()
        RGBLoss = input_img.grad.detach().clone()
        #del input_img
        input_img_wg = imageComposeMethod()
        
        #input_img_wg.backward(gradient=RGBLoss) 
        
        loss = torch.sum(input_img_wg * RGBLoss)
        
        if return_img:
            return loss,input_img_wg
        
        return loss
    
    # def IndirectForwardLoss(self,imageComposeMethod,content_image,print_loss=False,prev_image = None):
    #     with torch.no_grad():
    #         # 第一次前向不需要梯度
    #         input_img = imageComposeMethod() 

    #     # 第一次梯度计算
    #     input_img.requires_grad_(True)
    #     loss = self.evalLoss(input_img, content_image, print_loss)
    #     # 反向传播时释放计算图 (retain_graph=False)
    #     loss.backward(retain_graph=False)  
    #     # 分离并克隆梯度，之后立刻清除原梯度
    #     RGBLoss = input_img.grad.detach().clone()  
    #     input_img.grad = None  #  立即释放梯度缓存

    #     # 第二次前向需要重新生成 input_img（根据你的业务逻辑）
    #     # 如果不需要修改 input_img，可以复用之前的变量
    #     with torch.no_grad():
    #         input_img = imageComposeMethod() 

    #     # 关键优化点：直接传递梯度而非构建新计算图
    #     input_img.requires_grad_(True)
    #     # 等价于 torch.sum(input_img * RGBLoss).backward()
    #     # 但避免了创建中间计算图
    #     input_img.backward(gradient=RGBLoss)  
    

class NNFMstylizer(styleBase):
    def __init__(self,style_img,content_image,contentweight = 1,style_weight=100000):
        super().__init__(style_img,content_image)
        # cnn = models.vgg19(pretrained=True).features.eval()
        # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        # content_layers_default = ['conv_3','conv_4']
        # style_layers_default = [ 'conv_8','conv_9']

        # model, style_losses, content_losses = get_style_model_and_losses(cnn,
        #     style_img = style_img, content_img = content_image,
        #     content_layers = content_layers_default,style_layers=style_layers_default
            
        #     )

        # model.eval()
        # model.requires_grad_(False)

        self.style_weight = style_weight
        self.content_weight = contentweight
        self.style_img = style_img
        self.nnfm_loss_fn = nnfm_loss.NNFMLoss(device='cuda')

    def evalLoss(self,input_img,content_image,print_loss=False,prev_image = None):

        
        
        input_img = input_img

        
        loss_dict = self.nnfm_loss_fn(
        input_img,
        self.style_img,
        blocks=[
        2,
        ],
        loss_names=["nnfm_loss", "content_loss"],
        contents=content_image,
        )
        loss_dict["content_loss"] *=  10.0 *self.content_weight

        loss_dict["nnfm_loss"] = loss_dict["nnfm_loss"] * (self.style_weight * 10.0)

        loss = sum(list(loss_dict.values())) 
        

        
        if print_loss:
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                loss_dict["nnfm_loss"].item(), loss_dict["content_loss"].item()))
            print()
        
        



        return loss


class VGGstylizer(styleBase):
    def __init__(self,style_img,content_image,contentweight = 1,style_weight=1000000):
        super().__init__(style_img,content_image)
        cnn = models.vgg19(pretrained=True).features.eval()
        # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

        # content_layers_default = ['conv_6','conv_8']
        
        # content_layers_default = ['conv_4']
        # style_layers_default = [ 'conv_2']
        
        style_layers_default = [ 'conv_4','conv_5'] # adapted
        
        # style_layers_default = [ 'conv_2','conv_3']
        
        # # find good
        content_layers_default = ['conv_4']
        #style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        # get_style_model_and_losses_withModels_layerCount
    
        with torch.no_grad():
            self.model, self.style_losses, self.content_losses,self.content_models = get_style_model_and_losses_withModels(cnn,
            cnn_normalization_mean, cnn_normalization_std, style_img,content_layers_default, style_layers_default)
        
        # style_layers_default = [6]
        # with torch.no_grad():
        #     self.model, self.style_losses, self.content_losses,self.content_models = get_style_model_and_losses_withModels_layerCount(cnn,
        #     cnn_normalization_mean, cnn_normalization_std, style_img,content_layers_default, style_layers_default)


        self.model.eval()
        self.model.requires_grad_(False)
        self.style_weight = style_weight
        self.content_weight = contentweight
        self.style_img = style_img

    def evalLoss(self,input_img,content_image,print_loss=False,prev_image = None):
        
        style_score =0
        content_score = 0
        self.model(input_img)
        count =0 
        for sl in self.style_losses:
            style_score += sl.loss
            count = count+1
            
        count =0 
        for cl in self.content_losses:
            with torch.no_grad():
                feature_vector = self.content_models[count](content_image.to('cuda')).detach().clone()
            content_score += cl.loss(feature_vector)
            count = count+1
        
    
        # style_weight = 1000000

        # style_weight = 0
        # content_weight = 0
        
        # style_weight = 1000000
        # content_weight = 100
        
        
        content_score *= self.content_weight
        style_score *= self.style_weight

        
        
        loss = content_score + style_score
        
        if print_loss:
            print(f'current loss = {loss.item()}, current content_score = {content_score.item()}, current style_score = {style_score.item()}')

        return loss


    def evalVGGcontent(self,input_img,content_image,print_loss=False,prev_image = None):
        
 
        content_score = 0
        self.model(input_img)
        # exit(-1)
        count =0 
        for cl in self.content_losses:
            with torch.no_grad():
                feature_vector = self.content_models[count](content_image.to('cuda')).detach().clone()
                # print(feature_vector)
                
            content_score += cl.loss(feature_vector)
            count = count+1
        
    
        # style_weight = 1000000

        # style_weight = 0
        # content_weight = 0
        
        # style_weight = 1000000
        # content_weight = 100
        
        return content_score
        content_score *= self.content_weight
        style_score *= self.style_weight

        
        
        loss = content_score + style_score
        
        if print_loss:
            print(f'current loss = {loss.item()}, current content_score = {content_score.item()}, current style_score = {style_score.item()}')

        return loss

class ClipStylizer(styleBase):
    def __init__(self,style_img,content_image,contentweight=100,style_weight=1000, prompt='Sketch with black pencil',source_propt='A Photo'):
        super().__init__(style_img,content_image)
        self.prompt = 'Vangoghian style, swirling brushstrokes, thick impasto technique, vivid and contrasting colors'
        self.source_propt = source_propt
        
        device='cuda'
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.requires_grad_(False)
        

        self.template_text = compose_text_with_templates(self.prompt, imagenet_templates)
        self.template_source = compose_text_with_templates(self.source_propt, imagenet_templates)
        
        self.prepareTextLatent()
        ## not inited yet
        # self.source_features = None
        
        self.VGG = models.vgg19(pretrained=True).features
        self.VGG.to(device)

        for parameter in self.VGG.parameters():
            parameter.requires_grad_(False)
        
        self.cropper = transforms.Compose([
            transforms.RandomCrop(128 * 2)
        ])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
            transforms.Resize(224)
        ])
                
    def prepareTextLatent(self):
        with torch.no_grad():
            device='cuda'
            tokens = clip.tokenize(self.template_text).to(device)
            text_features = self.clip_model.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features
            
            # source embedding prompt
            
            tokens_source = clip.tokenize(self.template_source).to(device)
            text_source = self.clip_model.encode_text(tokens_source).detach()
            text_source = text_source.mean(axis=0, keepdim=True)
            text_source /= text_source.norm(dim=-1, keepdim=True)
            self.text_source = text_source
    
    def prepareContentImageLatent(self,content_image_ori):
        source_features = self.clip_model.encode_image(clip_normalize(content_image_ori,device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        self.source_features = source_features 
        self.content_features = get_features(img_normalize(content_image_ori), self.VGG)
        
    def evalLoss(self,input_img=None,content_image=None,print_loss=False,prev_image = None):

        # scheduler.step()
        # target = style_net(content_image,use_sigmoid=True).to(device)
        
        with torch.no_grad():
            self.prepareContentImageLatent(content_image)
        
        num_crops = 64
        target = input_img
        
        
        #input_here target
        target_features = get_features(img_normalize(target), self.VGG)
        
        content_loss = 0

        content_loss += torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - self.content_features['conv5_2']) ** 2)

        loss_patch=0 
        img_proc =[]
        for _ in range(num_crops):
            target_crop = self.cropper(target)
            target_crop = self.augment(target_crop)
            img_proc.append(target_crop)

        img_aug = torch.cat(img_proc,dim=0)

        
        image_features = self.clip_model.encode_image(clip_normalize(img_aug,device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        
        img_direction = (image_features - self.source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        text_direction = (self.text_features - self.text_source).repeat(image_features.size(0),1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        text_direction = text_direction
        
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
        
        
        loss_temp[loss_temp < 0.7] =0
        loss_patch += loss_temp.mean()
        
        reg_tv = (2e-3)*get_image_prior_losses_tv(target)
        # content_weight = 10
        total_loss = 9000*loss_patch + self.content_weight * content_loss * 1.01+ reg_tv
        
        
        if print_loss:
            print(f"Total loss: {total_loss.item()}, Content loss: {content_loss.item()}, Patch loss: {loss_patch.item()}, TV loss: {reg_tv.item()}")
            
        return total_loss




# class CLIPstylizer():
#     def __init__(self,style_img,content_image):
#         super().__init__(style_img,content_image)
#         cnn = models.vgg19(pretrained=True).features.eval()
#         # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
#         # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

#         content_layers_default = ['conv_3','conv_4']
#         style_layers_default = [ 'conv_8','conv_9']

#         with torch.no_grad():
#             self.model, self.style_losses, self.content_losses,self.content_models = get_style_model_and_losses_withModels(cnn,
#             cnn_normalization_mean, cnn_normalization_std, style_img,content_layers_default, style_layers_default)
#         self.model.eval()
#         self.model.requires_grad_(False)
#         self.style_weight = 1000000
#         self.content_weight = 100
#         self.style_img = style_img

#     def evalLoss(self,input_img,content_image,print_loss=False):

#         style_score =0
#         content_score = 0
#         self.model(input_img)
#         count =0 
#         for sl in self.style_losses:
#             style_score += sl.loss
#             count = count+1
            
#         count =0 
#         for cl in self.content_losses:
#             with torch.no_grad():
#                 feature_vector = self.content_models[count](content_image.to('cuda')).detach().clone()
#             content_score += cl.loss(feature_vector)
#             count = count+1
        
    
#         # style_weight = 1000000

#         # style_weight = 0
#         # content_weight = 0
        
#         # style_weight = 1000000
#         # content_weight = 100
        
        
#         content_score *= self.content_weight
#         style_score *= self.style_weight

        
        
#         loss = content_score + style_score
        
#         if print_loss:
#             print(f'current loss = {loss.item()}, current content_score = {content_score.item()}, current style_score = {style_score.item()}')

#         return loss


