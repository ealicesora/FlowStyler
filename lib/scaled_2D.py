import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

# from VGGUtils import *
# import imager_IO_utils as utils
from grid_sample_helper import *

import lib.Optical_Helper as OpticalHelper

class Field2D():
    def __init__(self, targetSize, scale = 1,downsample_ratio = 1,init_method="uv",channelcount = 2,initValue = None,enable_DownSample_beforeUse = False,DownSample_beforeUse_Ratio=2):
        self.originSize = targetSize
        B, C, H, W = targetSize
        
        self.enable_DownSample_beforeUse = enable_DownSample_beforeUse
        self.DownSample_beforeUse_Ratio = DownSample_beforeUse_Ratio
        
        self.scale = scale
        self.downsample_ratio = downsample_ratio
        
        # self.targetSize = targetSize
        H = H * scale // downsample_ratio
        W = W * scale // downsample_ratio
        output = None

        self.uv_base = None
        if init_method =='uv':
            xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1) 
            yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W) 
            xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
            yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
            grid = torch.cat((xx ,yy) ,1).float()
            grid = grid.cuda()

            vgrid = grid.clone()
            vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
            vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0
            vgrid = vgrid.permute(0 ,2 ,3 ,1)
            output = vgrid
            channelcount = 2
            self.uv_base = output.permute(0,3,1,2)
            
            output = torch.zeros([1,H,W,2])
        elif init_method =='zeros':
            output = torch.zeros([1,H,W,channelcount])
        elif init_method =='ones':
            output = torch.ones([1,H,W,channelcount])
        
        if initValue!=None:
            output[:,:,:]=initValue
        
        # self.output = output
        
        ## B, C, H, W
        # 使用 clone() 确保这是一个独立的张量而不是视图，避免 no_grad/grad 模式冲突
        self.output_ori = output.permute(0,3,1,2).clone()
        self.output = self.output_ori + 0.0
        
        self.channelcount = channelcount
        
        self.lr_scale = 1.0
        
        self.exp_avg = None
        self.exp_avg_sq = None
        
    def updateDataTemporally(self, updates):
        self.output = updates

    def updateDataForever(self, updates):
        self.output_ori = updates
        self.output = updates

    def UpdateAdamPara_fromOpt(self,optimizer,optical_frame = None,optical_frame_mask=None,no_advect = False):
        with torch.no_grad():
            cur_field_data = optimizer.state[self.output_ori]
            
            self.exp_avg = cur_field_data['exp_avg']
            self.exp_avg_sq = cur_field_data['exp_avg_sq']
            if no_advect:
                return
            self.exp_avg = self.advect_para_withOptical(self.exp_avg,optical_frame,paddingmode='zeros',sample_mode='bilinear') * optical_frame_mask
            self.exp_avg_sq = self.advect_para_withOptical(self.exp_avg_sq,optical_frame,paddingmode='zeros',sample_mode='bilinear') * optical_frame_mask

    def DumpPara_toOpt(self,optimizer):
        if self.exp_avg!=None:
            optimizer.state[self.output_ori]['exp_avg'] = self.exp_avg
            optimizer.state[self.output_ori]['exp_avg_sq'] = self.exp_avg_sq




    def getData(self):
        
        B, C, H, W = self.originSize
        
        if self.enable_DownSample_beforeUse:
            downsample_ratio = self.DownSample_beforeUse_Ratio
            
            # self.targetSize = targetSize
            H_d = H  // downsample_ratio
            W_d = W  // downsample_ratio
            
            UpUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            DownUpsampler = nn.Upsample(size = [H_d,W_d], mode='bilinear')
            res = DownUpsampler(self.output)
            res = UpUpsampler(res)
            
            return res
        
        # 完全断开并重新建立梯度连接以避免 no_grad/grad 模式冲突
        # detach() 断开所有历史，requires_grad_() 重新启用梯度追踪
        return self.output
        # result = self.output.detach()
        # if torch.is_grad_enabled():
        #     result.requires_grad_()
        # return result
    
    def get_colored_reshaped(self,size = None,colorOverwrite = None):
        
        targetData = self.getData()
        
        
        if size != None:
            B, C, H, W = size
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            # vgrid_new = output

            # vgrid_new = self.output.permute(0,3,1,2)
            # print(self.output.shape)
            # print([B,self.channelcount,H,W])
            if colorOverwrite != None:
                vgrid_new = OurUpsampler(colorOverwrite)
            else:
                vgrid_new = OurUpsampler(targetData)

        else:
            if colorOverwrite != None:
                vgrid_new = colorOverwrite
            else:
                vgrid_new = targetData
        
        
        alphaImage_ori = vgrid_new.permute(0,2,3,1)
        afterPermuteShape = list(tuple(alphaImage_ori.shape))
        alphaDim = self.channelcount
        alphaImage = alphaImage_ori.reshape([1,-1,alphaDim])        


        return alphaImage
    
    
    def get_colored(self,size = None,colorOverwrite = None):
        
        targetData = self.getData()
        
        
        if size != None:
            B, C, H, W = size
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            # vgrid_new = output

            # vgrid_new = self.output.permute(0,3,1,2)
            # print(self.output.shape)
            # print([B,self.channelcount,H,W])
            if colorOverwrite != None:
                vgrid_new = OurUpsampler(colorOverwrite)
            else:
                vgrid_new = OurUpsampler(targetData)

        else:
            if colorOverwrite != None:
                vgrid_new = colorOverwrite
            else:
                vgrid_new = targetData
                
        return vgrid_new
    
            # vgrid_new = vgrid_new.permute(0,2,3,1)      

    def get_colored_withUV(self,sample_uv,size = None):
        
        warped_result = grid_sample_custom(self.output, sample_uv,align_corners=False,mode='bilinear',padding_mode='border') 
        if size != None:
            B, C, H, W = size
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            # vgrid_new = output

            # vgrid_new = self.output.permute(0,3,1,2)
            # print(self.output.shape)
            # print([B,self.channelcount,H,W])
            vgrid_new = OurUpsampler(warped_result)
            return vgrid_new
        else:
            return self.output
            # vgrid_new = vgrid_new.permute(0,2,3,1)    

# with uv 


    def get_uv(self,size = None,warp_scale = 1.0):
        
        
        B, C, H, W = self.output.shape
        
        slice_x = self.output[:,0:1] / (H/2.0) 
        slice_y = self.output[:,1:2] / (W/2.0) 
        
        remapped = torch.cat([slice_x,slice_y],dim=1) * warp_scale
        
        
        if size != None:
            B, C, H, W = size
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')

            vgrid_new = remapped + self.uv_base
            vgrid_new = OurUpsampler(vgrid_new)

            vgrid_new = vgrid_new.permute(0,2,3,1)   
        else:
            vgrid_new = remapped + self.uv_base

            vgrid_new = vgrid_new.permute(0,2,3,1)   
        return vgrid_new
    
    def warp(self,target_img,warp_scale = 1.0):
        B, C, H, W = target_img.size()
        uv = self.get_uv(target_img.size(),warp_scale)
        # print(uv.shape)
        # print(uv.shape)
        
        # slice_x = uv[:,:,:,0:1] / (H/2.0)
        # slice_y = uv[:,:,:,1:2] / (W/2.0)
        
        # uv = torch.cat([slice_x,slice_y],dim=-1)
        # print(uv.shape)
        warped_result = grid_sample_custom(target_img, uv,align_corners=False,mode='bilinear',padding_mode='border') 
        return warped_result
    def backup_before_scale_weight(self):
        self.tensorBeforeUpdated = self.output_ori.detach().clone()

    def rescale_update_weight(self,masks,ratio):
        # mask 大部分区域是1 需要更新的地方是0
        with torch.no_grad():
            delta = self.output_ori - self.tensorBeforeUpdated
            self.output_ori -=  delta * masks * (1.0 - ratio)
        self.output_ori.requires_grad = True

    def prepare(self):
        # print(self.masks.shape)
        # print(self.p1.shape)
        # print(self.output_ori.shape)
        # self.output_ori = torch.where(self.masks > 0.5, self.p1, self.p2)
        self.output_ori = self.output_ori.detach().clone()
        self.output_ori.requires_grad = True
        
        self.output_ori[self.masks > 0.5] = self.p1
        self.output_ori[self.masks < 0.5] = self.p2
        
    def get_parameter_masks(self,masks):
        self.masks = masks
        with torch.no_grad():
            self.p1 = self.output_ori[masks > 0.5]
            self.p2 = self.output_ori[masks < 0.5]
        self.p1.requires_grad = True
        self.p2.requires_grad = True
        self.prepare()
        # normalize ``img``
        # self.output_ori.requires_grad = True
        return self.p1, self.p2
  
    
    def get_parameter(self):
        # normalize ``img``
        self.output_ori.requires_grad = True
        return [self.output_ori]
    
    def advect_withOptical(self,veloctiy_Field,backwardVel = None,paddingmode='zeros',sample_mode='billinear'):
        # should have the resolution of self.output
        
        B, C, H, W = self.output.shape
        
        if H!=veloctiy_Field.shape[2]:
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            veloctiy_Field_reshaped = OurUpsampler(veloctiy_Field.cuda().unsqueeze(0)).squeeze(0)
        else:
            veloctiy_Field_reshaped = veloctiy_Field.cuda()
        
        # reflection
        res = OpticalHelper.warp(self.output,veloctiy_Field_reshaped,padding_mode=paddingmode,enableClipping=True,sample_mode=sample_mode)
        return res

    def advect_para_withOptical(self,targetPara,veloctiy_Field,backwardVel = None,paddingmode='zeros',sample_mode='bilinear'):
        # should have the resolution of self.output
        
        B, C, H, W = self.output.shape
        
        if H!=veloctiy_Field.shape[2]:
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            veloctiy_Field_reshaped = OurUpsampler(veloctiy_Field.cuda().unsqueeze(0)).squeeze(0)
        else:
            veloctiy_Field_reshaped = veloctiy_Field.cuda()
        
        # reflection
        res = OpticalHelper.warp(targetPara,veloctiy_Field_reshaped,padding_mode=paddingmode,enableClipping=True,sample_mode=sample_mode)
        return res

    def translateShapeMatchThisField(self,target):
        B, C, H, W = self.output.shape
        if H!=target.shape[2]:
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            reshaped_target = OurUpsampler(target.unsqueeze(0)).squeeze(0)
        else:
            reshaped_target = target
        return reshaped_target
    def advect_withOptical_andUpdate(self,veloctiy_Field,backwardVel = None):
        
        B, C, H, W = self.output.shape
        
        if H!=veloctiy_Field.shape[2]:
            OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
            veloctiy_Field_reshaped = OurUpsampler(veloctiy_Field.cuda().unsqueeze(0)).squeeze(0)
        else:
            veloctiy_Field_reshaped = veloctiy_Field
        # 断开视图连接以避免 no_grad/grad 模式冲突
        output_detached = self.output#.detach().clone()
        res = OpticalHelper.warp(output_detached, veloctiy_Field_reshaped.cuda())
        self.output = res

    def reset(self):
        self.output = self.output_ori
        
  
    def advect(self,veloctiy_Field):
        B, C, H, W = self.originSize
        print('abandoned')
        exit('-1')
        
        # # self.targetSize = targetSize
        # H = H * self.scale // self.downsample_ratio
        # W = W * self.scale // self.downsample_ratio
        
        xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1) 
        yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W) 
        xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
        yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
        grid = torch.cat((xx ,yy) ,1).float()
        grid = grid.cuda()

        vgrid = grid.clone()
        vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
        vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0
        vgrid = vgrid.permute(0 ,2 ,3 ,1)
        base_grid = vgrid.permute(0,3,1,2)


        # print(veloctiy_Field.shape)
        # print(base_grid.shape)
        
        used_grid = base_grid + veloctiy_Field
        # print('---')
  
        used_grid = used_grid.permute(0 ,2 ,3 ,1)
   
        warped_result = grid_sample_custom(self.output, used_grid,align_corners=False,mode='bilinear',padding_mode='border') 
        
        
        H = H * self.scale // self.downsample_ratio
        W = W * self.scale // self.downsample_ratio
        OurUpsampler = nn.Upsample(size = [H,W], mode='bilinear')
        # vgrid_new = output

        # vgrid_new = self.output.permute(0,3,1,2)
        # print(self.output.shape)
        # print([B,self.channelcount,H,W])
        warped_result = OurUpsampler(warped_result)
        # print(warped_result.shape)
        return warped_result
    
        print(base_grid.shape)

# abandoned    
class Warped_Field2D():
    def __init__(self, colored,warp_field):
        self.colored = colored
        self.warp_field = warp_field

        
        ## B, H, W, C
        
    def get_warped(self,size = None):
        color = self.colored.get_colored(size)
        uv = self.warp_field.get_uv(size)
        
        warped_result = grid_sample_custom(color, uv,align_corners=False,mode='bilinear',padding_mode='border') 
        return warped_result
            # vgrid_new = vgrid_new.permute(0,2,3,1)      

    
    def get_parameter(self):
        # normalize ``img``
        return [self.color.get_parameter(),self.uv.get_parameter()]