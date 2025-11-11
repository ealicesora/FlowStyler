import torch






def match_colors_for_image_set_newShape(image_set, style_img):
    """
    image_set: [3,H, W ]
    style_img: [H, W, 3]
    """
    

    import copy
    style_img = copy.deepcopy(style_img)

    image_set = image_set.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set = image_set.reshape(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)#.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)
    

    return image_set


def match_colors_for_image_set_dNeRF(image_set, style_img, image_set_target_ori):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    image_set_target [N,3]
    """

    import copy
    style_img = copy.deepcopy(style_img)
    # print(image_set.shape)
    
    #print(image_set.shape)
    
    #print(image_set.shape)
    image_set = image_set.squeeze(0)#.permute(1,2,0)
    image_set = image_set.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
   
    # image_set_target = image_set_target.squeeze(0)#.permute(1,2,0)
    # image_set_target = image_set_target.permute(1, 2, 0)
    # image_set_target = image_set_target.unsqueeze(0)
    
    style_img = style_img.squeeze(0)
    style_img = style_img.permute(1, 2, 0)
    
    
    sh = image_set.shape
    image_set_target = image_set_target_ori.clone()
    image_set_target = image_set_target.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set_target.device)

    mu_c = image_set_target.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set_target - mu_c).transpose(1, 0), image_set_target - mu_c) / float(image_set_target.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)

    return image_set, color_tf



def match_colors_for_image_set_newShape_dNeRF(image_set, style_img,image_set_target):
    """
    image_set: [3,H, W ]
    style_img: [H, W, 3]
    """
    

    import copy
    style_img = copy.deepcopy(style_img)

    image_set = image_set.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set_target = image_set_target.reshape(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set_target.device)

    mu_c = image_set_target.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set_target - mu_c).transpose(1, 0), image_set_target - mu_c) / float(image_set_target.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)#.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)
    

    return image_set





def match_colors_for_image_set(image_set_ori, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """

    import copy
    style_img = copy.deepcopy(style_img)
    # print(image_set.shape)
    image_set = image_set_ori.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    #print(image_set.shape)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    # print('scl_c')
    # print(scl_c)
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))
    # print('scl_s')
    # print(scl_s)
    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)

    LerpScale = 1.0
    if LerpScale != 1.0:
        image_set = image_set * LerpScale + (1.0 - image_set) * image_set_ori
    
    return image_set, color_tf


def match_colors_for_image_set_ds(image_set_ori, style_img, alpha=0.8, beta=0.1):
    """
    Args:
        alpha (float): 风格化强度控制 (0.0~1.0)
        beta (float): 协方差正则化强度 (防止数值不稳定)
    """
    device = image_set_ori.device
    
    # 预处理张量维度
    image_set = image_set_ori.clone().squeeze(0).permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]
    style_img = style_img.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
    
    # 展平为像素点集合
    sh = image_set.shape
    image_flat = image_set.view(-1, 3)
    style_flat = style_img.view(-1, 3).to(device)
    
    # 计算均值
    mu_c = image_flat.mean(0, keepdim=True)
    mu_s = style_flat.mean(0, keepdim=True)
    
    # 计算协方差矩阵（带正则化）
    eps = beta * torch.eye(3, device=device)  # 正则化项
    cov_c = (image_flat - mu_c).T @ (image_flat - mu_c) / image_flat.shape[0] + eps
    cov_s = (style_flat - mu_s).T @ (style_flat - mu_s) / style_flat.shape[0] + eps
    
    # SVD分解
    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)
    
    # 特征值截断（控制最大缩放幅度）
    max_eigenvalue = 10.0  # 限制最大特征值避免过度拉伸
    sig_c_clamped = torch.clamp(sig_c, min=1e-6, max=max_eigenvalue)
    sig_s_clamped = torch.clamp(sig_s, min=1e-6, max=max_eigenvalue)
    
    # 构造缩放矩阵
    scl_c = torch.diag(1.0 / torch.sqrt(sig_c_clamped))
    scl_s = torch.diag(torch.sqrt(sig_s_clamped))
    
    # 核心变换矩阵（加入alpha控制强度）
    tmp_mat = alpha * (u_s @ scl_s @ u_s.T @ u_c @ scl_c @ u_c.T)
    tmp_vec = mu_s - alpha * (mu_c @ tmp_mat.T)
    
    # 应用变换
    transformed = image_flat @ tmp_mat.T + tmp_vec
    
    # 动态范围缩放（替代硬截断）
    # min_val = transformed.min(dim=0).values
    # max_val = transformed.max(dim=0).values
    # transformed = (transformed - min_val) / (max_val - min_val + 1e-8)
    
    # 恢复张量形状
    output = transformed.view(sh).permute(0, 3, 1, 2)  # [1, 3, H, W]
    
    # 可选：与原图线性混合（进一步控制风格化程度）
    if alpha < 1.0:
        output = alpha * output + (1 - alpha) * image_set_ori
        
    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return output.clamp(0.0, 1.0), color_tf

def apply_CT(image_set,colorTransfer):
    tmp_mat = colorTransfer[:3, :3]
    tmp_vec = colorTransfer[:3, 3:4] 
    
    image_set = image_set.squeeze(0).permute(1,2,0)
    
    sh = image_set.shape
    
    image_set = image_set.view(-1, 3)
    determinants = torch.linalg.det(tmp_mat)
    # print(determinants)
    # # image_set = image_set + 1.0* torch.tensor([0.2,0.02,0.2]).view(1, 3)
    # print(tmp_vec)
    # tmp_vec[0] = 0.0
    # tmp_vec[1] = 0.0
    #image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)#/3.0
    # print('cd num')
    # condition_numbers = torch.linalg.cond(tmp_mat, p='fro')
    # print(condition_numbers)
    
    # image_set = image_set @ tmp_mat.T + 1.0* torch.tensor([0.0,0.,0.0]).view(1, 3)
    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().view(sh)
    image_set = image_set.permute(2,0,1).unsqueeze(0)

    # image_set = torch.sigmoid(image_set)
    
    return image_set