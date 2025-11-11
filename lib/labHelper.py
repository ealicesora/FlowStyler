import torch
import cv2
import numpy as np

def tensor_rgb_to_lab(tensor):
    """ 将RGB张量转换为LAB张量
    Args:
        tensor: [B, C, H, W] 或 [H, W, C], 值域[0,1]
    Returns:
        LAB张量，形状同输入，L通道[0,100]，a/b通道[-128,127]
    """
    # 转换到numpy并调整范围

    np_img = tensor.cpu().numpy().transpose(0, 2, 3, 1) if tensor.dim() == 4 else tensor.cpu().numpy()
    # np_img =  tensor.squeeze(0).cpu().numpy()
    np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
    
    # 使用OpenCV转换颜色空间
    # lab_imgs = []
    # for img in np_img:
    #     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #     lab_imgs.append(lab)

    lab_np = cv2.cvtColor(np_img[0], cv2.COLOR_RGB2LAB)
    
    # 转换为张量并调整维度
    lab_tensor = torch.from_numpy(lab_np).to(tensor.device).float().unsqueeze(0)
    lab_tensor = lab_tensor.permute(0, 3, 1, 2) if tensor.dim() == 4 else lab_tensor.permute(2, 0, 1)
    
    # 标准化LAB通道
    lab_tensor[..., 0, :, :] /= 2.55   # L通道 [0,100] => [0,100]/2.55 => [0,39.2]
    lab_tensor[..., 1:, :, :] -= 128   # a,b通道 [-128,127] => [0,255]-128
    return lab_tensor

def tensor_lab_to_rgb(tensor):
    """ 将LAB张量转回RGB张量
    Args:
        tensor: LAB张量，形状[B, C, H, W]或[H, W, C]
    Returns:
        RGB张量，值域[0,1]
    """
    # 反标准化LAB通道
    device = tensor.device
    tensor = tensor.detach().clone()
    if tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
        tensor[..., 0] *= 2.55    # L通道 [0,100]
        tensor[..., 1:] += 128.0 
    else:
        tensor = tensor.permute(1, 2, 0)
        tensor[..., 0] *= 2.55
        tensor[..., 1:] += 128.0 
    
    # 转换到numpy
    np_lab = tensor.cpu().numpy().astype(np.uint8)
    
    # 使用OpenCV转换颜色空间
    # rgb_imgs = []
    # for lab in np_lab:
    #     rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    #     rgb_imgs.append(rgb)
    # rgb_np = np.stack(rgb_imgs, axis=0)

    rgb_np = cv2.cvtColor(np_lab.squeeze(0), cv2.COLOR_LAB2RGB)
    
    # 转换为张量并标准化
    rgb_tensor = torch.from_numpy(rgb_np).to(device).float() / 255.0
    return rgb_tensor.unsqueeze(0).permute(0, 3, 1, 2) if tensor.dim() == 4 else rgb_tensor.permute(2, 0, 1)

def match_colors_lab_cv2(image_set_ori, style_img, alpha=0.8, beta=0.1):
    """
    使用OpenCV颜色转换的LAB空间版本
    """
    device = image_set_ori.device
    
    # 转换到LAB空间
    image_lab = tensor_rgb_to_lab(image_set_ori.unsqueeze(0))  # [1, 3, H, W]
    style_lab = tensor_rgb_to_lab(style_img)       # [3, H, W]
    
    # print(style_img.shape)
    # print(image_set_ori.shape)
    # print('-----')
    
    # 预处理维度
    image_lab = image_lab.permute(0, 2, 3, 1)  # [1, H, W, 3]

    style_lab = style_lab.squeeze(0).permute(1, 2, 0)      # [H, W, 3]
    # print(style_lab)
    # 展平数据
    sh = image_lab.shape
    image_flat = image_lab.view(-1, 3)
    style_flat = style_lab.view(-1, 3)
    
    # 计算统计量
    mu_c = image_flat.mean(0, keepdim=True)
    mu_s = style_flat.mean(0, keepdim=True)
    
    # 协方差计算（带正则化）
    eps = beta * torch.eye(3, device=device)
    cov_c = (image_flat - mu_c).T @ (image_flat - mu_c) / image_flat.shape[0] + eps
    cov_s = (style_flat - mu_s).T @ (style_flat - mu_s) / style_flat.shape[0] + eps
    
    # SVD分解
    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)
    
    # 特征值截断
    max_eigenvalue = 10.0
    sig_c_clamped = torch.clamp(sig_c, min=1e-6, max=max_eigenvalue)
    sig_s_clamped = torch.clamp(sig_s, min=1e-6, max=max_eigenvalue)
    
    # 构造变换矩阵
    scl_c = torch.diag(1.0 / torch.sqrt(sig_c_clamped))
    scl_s = torch.diag(torch.sqrt(sig_s_clamped))
    tmp_mat = alpha * (u_s @ scl_s @ u_s.T @ u_c @ scl_c @ u_c.T)
    tmp_vec = mu_s - alpha * (mu_c @ tmp_mat.T)
    
    # 应用变换
    transformed = image_flat @ tmp_mat.T + tmp_vec
    
    # print(transformed.shape)
    # print(image_flat.shape)
    
    transformed = style_flat
    
    transformed[:,0] = image_flat[:,0]
    
    # 重建LAB图像
    output_lab = transformed.view(sh)
    
    #output_lab = image_flat.view(sh)
    # 转换回RGB
    output_rgb = tensor_lab_to_rgb(output_lab.permute(0, 3, 1, 2))
    
    # 动态范围调整
    # output_flat = output_rgb.reshape(-1, 3)
    # min_val = output_flat.min(dim=0).values
    # max_val = output_flat.max(dim=0).values
    # output_rgb = (output_rgb - min_val) / (max_val - min_val + 1e-8)
    
    # 与原图混合
    if alpha < 1.0:
        output_rgb = alpha * output_rgb + (1 - alpha) * image_set_ori
    
    # 构造变换矩阵
    color_tf = torch.eye(4).float().to(device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    return output_rgb, color_tf
