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
from lib.MatrixHelper import *

def downSampleAndUpsample(target):
    B, C, H, W = target.shape
    OurDownsampler = nn.Upsample(size = [H//4,W//4], mode='bicubic')
    # vgrid_new = output

    # vgrid_new = self.output.permute(0,3,1,2)
    # print(self.output.shape)
    # print([B,self.channelcount,H,W])
    target = OurDownsampler(target)
    
    OurUpsampler = nn.Upsample(size = [H,W], mode='bicubic')
    target = OurUpsampler(target)
    return target


def particle2Image_cubic(p_p,color,vox_size,kernel_size = 4,effectRangeSize = 4,kernel_method='cubic'):
    # p_p color 1,N,1/3
    pc = p_p
    bs = pc.size(0)
    n = pc.size(1)
    # eps=1e-6
    grid = (pc + 1.0) * (vox_size)/2.0
    grid_floor = grid.floor()
    grid_idxs = grid_floor.long()
    batch_idxs = torch.arange(bs)[:, None, None].repeat(1, n, 1).to(pc.device)

    idxs = torch.cat([batch_idxs, grid_idxs], dim=-1).view(-1, 3)
  
    r = grid - grid_floor
    # print(vox_size)

    xsize = int(vox_size[0].item())
    ysize = int(vox_size[1].item())
    # print(xsize)
    # print(ysize)
    

    voxels_color_slice = pc.new(bs, xsize, ysize,3).fill_(0)
    voxels_density = pc.new(bs, xsize, ysize,1).fill_(0)

    
    def interp(pos):
        
        from lib.kernelHelper import KernelMethod

        update = KernelMethod(-r[...,0] + pos[0],-r[...,1] + pos[1],kernel_size,kernel_method) 

        update = update.unsqueeze(-1)
        update_col = color * update
        
        
        shift_idxs = torch.LongTensor([[0] + pos]).to(pc.device)
        shift_idxs = shift_idxs.repeat(idxs.size(0), 1)
        update_idxs = idxs + shift_idxs
        # print(update_idxs)
        valid_new_x_0 = update_idxs[...,1] >=0 
        valid_new_y_0 = update_idxs[...,2] >=0
        valid_new_0 = valid_new_x_0 & valid_new_y_0

        valid_new_x_end = update_idxs[...,1] < xsize
        valid_new_y_end = update_idxs[...,2] < ysize
        valid_new_end = valid_new_x_end & valid_new_y_end    
        
        valid_new = valid_new_0 & valid_new_end
        # print(update)
        update = update.view(-1,1)[valid_new]

        update_col = update_col.view(-1,3)[valid_new]

        
        update_idxs = update_idxs[valid_new]
           

        voxels_color_slice.index_put_(torch.unbind(update_idxs, dim=1), update_col, accumulate=True)
        voxels_density.index_put_(torch.unbind(update_idxs, dim=1), update, accumulate=True)

    addition_grid_size = effectRangeSize -1
    for k in range(-addition_grid_size,2 + addition_grid_size):
        for j in range(-addition_grid_size,2 + addition_grid_size):
            # if k==0 and j==0:
                interp([k, j])

    res = voxels_color_slice
    res_d = voxels_density
         

    res = res/(res_d + 0.00001)
    return res


class Larg_Image():
    def __init__(self, colorImage, downSample_ratio = 2):
        # colorImage 1,H,W,3
        self.original_shape = colorImage.shape
        B, C, H, W = colorImage.shape
        
        self.downSample_ratio = downSample_ratio
        OurDownsampler = nn.Upsample(size = [H//self.downSample_ratio,W//self.downSample_ratio], mode='bilinear')
        # vgrid_new = output

        # vgrid_new = self.output.permute(0,3,1,2)
        # print(self.output.shape)
        # print([B,self.channelcount,H,W])
        colorImage = OurDownsampler(colorImage)
        
        # OurUpsampler = nn.Upsample(size = [H,W], mode='bicubic')
        # colorImage = OurUpsampler(colorImage)

        
        self.downsampledShape = colorImage.shape
        color_field = colorImage.squeeze(0).permute(1,2,0).unsqueeze(0).cuda()
        color_field = color_field.reshape([1,-1,3])
        self.color_field = color_field.cuda()
        
        
        self.vgrid_ndc = make_grids_no_allign_centers(colorImage,scale=1,downSampleScale = 1).cuda()
        self.vgrid_ndc = self.vgrid_ndc
        self.vgrid_ndc = self.vgrid_ndc.reshape([1,-1,2])
        import copy
        # self.vgrid_ndc_copied = copy.deepcopy(vgrid_ndc) 
        # vgrid_ndc.shape

        # init_pos = vgrid_ndc

        self.currentSize = colorImage.shape
        
        xsize = self.original_shape[-1]
        ysize =  self.original_shape[-2]

        # size = 64
        self.vox_size = color_field.new(2)
        self.vox_size[0] = xsize
        self.vox_size[1] = ysize
        # print(self.vox_size)
        self.WarpingFieldSampleUV = self.getImageSampleUV().cuda()

        # self.vgrid_ndc
        
    def setNewImage(self, new_colorImage):
        # self.original_shape = colorImage.shape
        B, C, H, W = self.original_shape
        OurDownsampler = nn.Upsample(size = [H//self.downSample_ratio,W//self.downSample_ratio], mode='bilinear')
        # vgrid_new = output

        # vgrid_new = self.output.permute(0,3,1,2)
        # print(self.output.shape)
        # print([B,self.channelcount,H,W])
        colorImage = OurDownsampler(new_colorImage)
        
        # OurUpsampler = nn.Upsample(size = [H,W], mode='bicubic')
        # colorImage = OurUpsampler(colorImage)
    
        color_field = colorImage.squeeze(0).permute(1,2,0).unsqueeze(0).cuda()
        color_field = color_field.reshape([1,-1,3])
        self.color_field = color_field

    # aborted    
    # def getImage(self, targetSize = None, kernel_size = 4,effectRangeSize = 4,kernel_method='cubic',upsampleMethod = 'bilinear'):
    #     raw_image = particle2Image_cubic(self.vgrid_ndc,self.color_field,self.vox_size,kernel_size = kernel_size,effectRangeSize = effectRangeSize,kernel_method=kernel_method)
    #     raw_image_reshape = raw_image.permute(0,3,2,1)
    #     # self.original_shape = colorImage.shape
    #     if targetSize == None:
    #         targetSize = self.original_shape
    #     B, C, H, W = targetSize
    #     OurUpsampler = nn.Upsample(size = [H,W], mode=upsampleMethod)
    #     raw_image_reshape_resized = OurUpsampler(raw_image_reshape)   
        
    #     return raw_image_reshape_resized
    
    
    def getAlphaComposedImage(self,alpha_image,alphamode,alphahelper = None,getMatrixOnly =False,only_alpha= False):
        used_field = self.color_field
        
        if alphahelper!=None:
            # print(alphahelper.getData().shape)
            Identity33 = torch.tensor([1.0,0.0,0.0,  0.0,1.0,0.0,  0.0,0.0,1.0], device=used_field.device)
            alpha_image_used_helper = alphahelper.getData()  
            alpha_image_used_helper = alpha_image_used_helper.squeeze(0).permute(1,2,0).reshape([-1,9])# +Identity33.unsqueeze(0).unsqueeze(0)                
            matrices_helper = alpha_image_used_helper.reshape(-1, 3, 3)    
        
        if alpha_image !=  None:
            if alphamode == 'alphamode':
                # used_field = used_field * (alpha_image + 1.0)
                if only_alpha:
                    return alpha_image
                used_field = used_field + (alpha_image )
            elif alphamode == 'Matrix33':
                
                # print(used_field.shape)
                
                padding  = torch.tensor([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0], device=alpha_image.device)
                alpha_image_used = alpha_image[...,:] + padding                          
                matrices = alpha_image_used.view(-1, 3, 3)


                if getMatrixOnly:
                    return matrices
                vectors = used_field.unsqueeze(-1).squeeze(0)

                used_field = torch.bmm(matrices, vectors).unsqueeze(0).squeeze(-1)
                
            elif alphamode == 'Simple_adding':
                
                # print(used_field.shape)
                
                #padding  = torch.tensor([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
                alpha_image_used = alpha_image# [...,:] #+ padding                          
                # matrices = alpha_image_used#.reshape(-1, 3, 3)


                vectors = used_field.squeeze(0)

                
                matrices = alpha_image_used.squeeze(0)
                adding = matrices[:,:3]
            
                if getMatrixOnly:
                    return matrices
                used_field  = vectors + adding
                # print(vectors.shape)
                # print(matrices.shape)
                
                # used_field = torch.bmm(matrices, vectors).unsqueeze(0).squeeze(-1)
                
  

            elif alphamode == 'Rot33':
                
                # print(used_field.shape)
                
                padding  = torch.tensor([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0], device=alpha_image.device)
                alpha_image_used = alpha_image[...,:9] + padding                          
                matrices = alpha_image_used.view(-1, 3, 3)

                matrices = fully_diff_rotation(matrices)

                if getMatrixOnly:
                    return matrices                

                vectors = used_field.unsqueeze(-1).squeeze(0)

                used_field = torch.bmm(matrices, vectors).unsqueeze(0).squeeze(-1)
                
            
            elif alphamode == 'Matrix44':
                
                vectors = used_field.squeeze(0)
                # print(vectors.shape)
                padding = torch.tensor([1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0,   0.0,0.0,0.0,1.0], device=alpha_image.device)
                # print(padding.shape)
                alpha_image_used = alpha_image[...,:] + padding
                n = vectors.shape[0]
                # print(vectors.shape)
                vectors = torch.cat([vectors, torch.ones([n, 1], device=vectors.device)], dim=1).unsqueeze(-1)
                matrices = alpha_image_used.view(-1, 4, 4)
                # 进行矩阵乘法实现仿射变换

                if getMatrixOnly:
                    return matrices

                # print(matrices.shape)
                # print(vectors.shape)
                used_field = torch.bmm(matrices, vectors)

                # 将结果压缩回n,3的形状
                used_field = used_field.squeeze(-1)[:, :3]

            elif alphamode == 'Scaled_Exp_Matrix':       
                # print(alpha_image.shape)
                B,H,_ = alpha_image.shape
                
                params = alpha_image

                
                theta = params[..., :3]  # 旋转参数
                s = params[..., 3:6]     # 缩放参数

                # 构建反对称矩阵
                A = torch.zeros(H, 3, 3, device=alpha_image.device)
                A[..., 0, 1] = theta[..., 0]
                A[..., 0, 2] = theta[..., 1]
                A[..., 1, 2] = theta[..., 2]
                A = A - A.transpose(-1, -2)  # 确保反对称

                # 矩阵指数
                R = torch.matrix_exp(A)

                # 构建缩放矩阵
                S = torch.diag_embed(torch.exp(s))
                sclaed_matrix = S @ R
                
                if getMatrixOnly:
                    return sclaed_matrix
                
                if alphahelper!=None:
                    matrix = sclaed_matrix + matrices_helper
                
                vectors = used_field.squeeze(0).squeeze(0)
                # print(vectors.shape)
                # print(sclaed_matrix.shape)
                used_field = torch.bmm(sclaed_matrix.squeeze(0), vectors.unsqueeze(-1)).unsqueeze(0).squeeze(-1)
                
            elif alphamode == 'Quat_Matrix':       
                # print(alpha_image.shape)
                B,H,_ = alpha_image.shape
                
                params = alpha_image

                


                w, x, y, z = params[...,:4].unbind(-1)
                
                norm_factor = w*w + x*x + y*y + z*z + 1e-5
                
                norm_factor = torch.sqrt(norm_factor)
                
                w = w/norm_factor
                x = x/norm_factor
                y = y/norm_factor
                z = z/norm_factor
                
                # w = w+1.0
                matrix = torch.stack([
                    1.0-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w,
                    2*x*y+2*z*w,   1.0-2*x*x-2*z*z,   2*y*z-2*x*w,
                    2*x*z-2*y*w,     2*y*z+2*x*w, 1.0-2*x*x-2*y*y
                ], dim=-1).view(-1, 3, 3)
                
                # print(matrix[0])
                
                #print(matrix[0])
                
                if getMatrixOnly:
                    return matrix
                
                if alphahelper!=None:
                    1
                    # print(matrices_helper.shape)
                    # print(matrix.shape)
                    # print(matrix[0])
                    # print(matrices_helper[0])
                    matrix = torch.bmm(matrix, matrices_helper)

                    
                vectors = used_field.squeeze(0).squeeze(0)
                # print(vectors.shape)
                # print(sclaed_matrix.shape)
                used_field = torch.bmm(matrix.squeeze(0), vectors.unsqueeze(-1)).unsqueeze(0).squeeze(-1)


            elif alphamode == 'Quat_Matrix_Naive':       
                # print(alpha_image.shape)
                B,H,_ = alpha_image.shape
                
                params = alpha_image

                


                w, x, y, z = params[...,:4].unbind(-1)
                
                w = torch.abs(w+1.0)
                
                
                # norm_factor = w*w + x*x + y*y + z*z + 1e-5
                
                # norm_factor = torch.sqrt(norm_factor)
                
                # w = w/norm_factor
                # x = x/norm_factor
                # y = y/norm_factor
                # z = z/norm_factor
                
                # w = w+1.0
                matrix = torch.stack([
                    1.0-2*y*y-2*z*z,   2*x*y-2*z*w,   2*x*z+2*y*w,
                    2*x*y+2*z*w,   1.0-2*x*x-2*z*z,   2*y*z-2*x*w,
                    2*x*z-2*y*w,     2*y*z+2*x*w, 1.0-2*x*x-2*y*y
                ], dim=-1).view(-1, 3, 3)
                
                # print(matrix[0])
                
                #print(matrix[0])
                
                if getMatrixOnly:
                    return matrix
                
                if alphahelper!=None:
                    1
                    # print(matrices_helper.shape)
                    # print(matrix.shape)
                    # print(matrix[0])
                    # print(matrices_helper[0])
                    matrix = torch.bmm(matrix, matrices_helper)

                    
                vectors = used_field.squeeze(0).squeeze(0)
                # print(vectors.shape)
                # print(sclaed_matrix.shape)
                used_field = torch.bmm(matrix.squeeze(0), vectors.unsqueeze(-1)).unsqueeze(0).squeeze(-1)
      
            elif alphamode == 'PI_Matrix':       
                # print(alpha_image.shape)
                B,H,_ = alpha_image.shape
                
                params = alpha_image
                
                params[...,0] +=0.5413
                params[...,2] +=0.5413
                params[...,5] +=0.5413
                # params = params/1.7
                matrix = build_scaled_spd_matrix(params)
                # print(matrix[0])
                # # exit(-1)

                if getMatrixOnly:
                    return matrix

                if alphahelper!=None:
                    matrix = matrix + matrices_helper
                    
                vectors = used_field.squeeze(0).squeeze(0)
                # print(vectors.shape)
                # print(sclaed_matrix.shape)
                used_field = torch.bmm(matrix.squeeze(0), vectors.unsqueeze(-1)).unsqueeze(0).squeeze(-1)                
           
        return used_field
    
    
    def getMatrixRegLoss(self,alpha_image,alphamode,print_loss=False,weight = 1.0):
        matrix = self.getAlphaComposedImage(alpha_image = alpha_image,alphamode = alphamode,getMatrixOnly = True)
        if alphamode == 'alphamode':
            return torch.tensor(0.0, device=self.color_field.device)
        
        if alphamode != 'Matrix33' and alphamode!='Scaled_Exp_Matrix':
            return torch.tensor(0.0, device=self.color_field.device)
        
        # return torch.tensor(0.0)
        if print_loss:
            print('matrix loss')
        
        lambda_reg = 1e8 * weight
        # loss 1
        # ortho_loss = torch.norm(matrix @ matrix.transpose(-2, -1) - torch.eye(3), p='fro')**2
        ortho_loss = torch.mean((matrix @ matrix.transpose(-2, -1) - torch.eye(3, device=matrix.device)) ** 2) * lambda_reg
        
        total_loss = ortho_loss
        if print_loss:
            print(total_loss)
        # loss 2
        det = torch.det(matrix)  # 计算行列式
        epsilon = 0.2
        reg_loss_min = torch.mean(torch.clamp(epsilon - torch.abs(det), min=0)**2) * lambda_reg
        if print_loss:    
            print(reg_loss_min)
        
        epsilon_max = 1.5
        reg_loss_max = torch.mean(torch.clamp(torch.abs(det) - epsilon_max, min=0)**2) * lambda_reg
        if print_loss:
            print(reg_loss_max)    
    
        total_loss +=  reg_loss_min + reg_loss_max 
        
        
        
        # def manual_cond(matrix, p='fro'):
        #     if p == 'fro':
        #         # 计算 Frobenius 范数条件数
        #         U, S, V = torch.linalg.svd(matrix)
        #         cond = S[0] / S[-1]
        #     else:
        #         raise NotImplementedError
        #     return cond
        # condition_numbers = manual_cond(matrix, p='fro')
        # condition_loss = torch.mean(condition_numbers)* lambda_reg
        # if print_loss:
        #     print(condition_loss)    
        # total_loss +=  condition_loss  
        
        
        # print(matrix.shape)
        return total_loss 
    
    
    def getImage_byNewposition(self,new_pos, targetSize = None, kernel_size = 4,effectRangeSize = 4,kernel_method='cubic',upsampleMethod = 'bilinear',alpha_image = None,alphamode='alphamode',alphaHelper = None):


        used_field = self.getAlphaComposedImage(alpha_image,alphamode,alphaHelper)

            
            
        # print(used_field.shape)
        # print(alpha_image.shape)
        # print(self.color_field.shape)
        raw_image = particle2Image_cubic(new_pos,used_field,self.vox_size,kernel_size = kernel_size,effectRangeSize = effectRangeSize,kernel_method=kernel_method)

        raw_image_reshape = raw_image.permute(0,3,2,1)

        if targetSize == None:
            targetSize = self.original_shape
        B, C, H, W = targetSize
        OurUpsampler = nn.Upsample(size = [H,W], mode=upsampleMethod)

        raw_image_reshape_resized = OurUpsampler(raw_image_reshape)   
        
        return raw_image_reshape_resized
    
    
    # def getParameter(self):
    #     return self.vgrid_ndc
    
    def getImageSampleUV(self):
        B, C, H, W = self.currentSize
        remappedUV = torch.zeros_like(self.vgrid_ndc)
        remappedUV[..., 0] = (self.vgrid_ndc[..., 0] + 1.0) * W / ( W - 1) - 1.0
        remappedUV[..., 1] = (self.vgrid_ndc[..., 1] + 1.0) * H / ( H - 1) - 1.0
        return remappedUV
    
    # aborted cause no scaling
    # def getUpdatedParticlePostionFromWarpingField(self,warpingField):

    #     uv_used = self.WarpingFieldSampleUV.unsqueeze(0)
        
    #     warped_result = grid_sample_custom(warpingField, uv_used,align_corners=False,mode='bicubic',padding_mode='border') 

    #     warped_result = warped_result.reshape([1,2,-1]).permute(0,2,1)
        
    #     res = warped_result + self.vgrid_ndc 
        
    #     return res

          
    def getUpdatedParticlePostionFromWarpingField(self,warpingField,addi_mul = 1.0):
        # print(warpingField.shape)
        
        # addi_mul = 10.0
        
        B, C, H, W = self.currentSize
        
        slice_x = warpingField[:,0:1] / (H/2.0) * addi_mul
        slice_y = warpingField[:,1:2] / (W/2.0) * addi_mul
        
        remapped = torch.cat([slice_x,slice_y],dim=1)
        
        uv_used = self.WarpingFieldSampleUV.unsqueeze(0)
        
        warped_result = grid_sample_custom(remapped, uv_used,align_corners=False,mode='bilinear',padding_mode='border') 

        
        # print(warpingField.shape)
        # print(uv_used.shape)
        
        
        warped_result = warped_result.reshape([1,2,-1]).permute(0,2,1)
        # print(warped_result.shape)
        
        # exit(-1)
        res = warped_result + self.vgrid_ndc 
        
        return res



    def getValueFromField(self, warpingField):
        
        raise NotImplementedError('should not reached')
        uv_used = self.WarpingFieldSampleUV.unsqueeze(0)

        warped_result = grid_sample_custom(warpingField, uv_used,align_corners=False,mode='bicubic',padding_mode='bilinear') 

        warped_result = warped_result.reshape([1,2,-1]).permute(0,2,1)
        
        res = warped_result #+ self.vgrid_ndc 
        
        return res
    

    def getColorFromSamplePostition(self,colorField):
        uv_used = self.WarpingFieldSampleUV.unsqueeze(0)

        colorField.permute(0,1,3,2)
        
        warped_result = grid_sample_custom(colorField, uv_used,align_corners=False,mode='bilinear',padding_mode='border') 

        warped_result = warped_result.reshape([1,3,-1]).permute(0,2,1)
        
        # res = warped_result + self.vgrid_ndc 
        
        return warped_result
        
    
    
        