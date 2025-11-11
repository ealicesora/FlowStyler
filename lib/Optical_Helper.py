import numpy as np
import torch

import torch.nn.functional as F

import os
# making no connection
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def alignToDivable(image1,padding_factor = 32):
    
    
    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    # resize to nearest size or specified size
    inference_size = nearest_size 
    

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        
    return image1


def warp(x, flo,padding_mode='zeros', enableClipping = False,sample_mode='bilinear'):
    DEVICE = 'cuda'
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    
    if enableClipping:
        vgrid[:, 0, :, :] = vgrid[:, 0, :, :].clip(0.0,W - 1)
        vgrid[:, 1, :, :] = vgrid[:, 1, :, :].clip(0.0,H - 1)
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    
    # not sured
    if enableClipping:
        vgrid = vgrid.clip(-1.0+1e-10,1.0-1e-10)
    
    
    import torch.nn.functional as TF
    vgrid = vgrid.permute(0, 2, 3, 1)
    # this is important and test needed align corner
    output = TF.grid_sample(x, vgrid,padding_mode=padding_mode,mode=sample_mode,align_corners= True)
    
    # mask = torch.ones(x.size()).to(DEVICE)
    # mask = TF.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    return output

import torchvision.transforms as T

RAFT_PREPROCESS = T.Compose(
    [
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
    ]
)



def GetRaftModel():
    global RaftModel
    return RaftModel

def MoveModel(dest):
    RaftModel.to(dest)

from torchvision.models.optical_flow import raft_large

def GetRaftModel():
    # model = raft_large(pretrained=True, progress=False).to('cuda')
    # model = model.eval()
    # RaftModel = model
    # return RaftModel
    ckpt_path = "./models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

    model = raft_large(weights=None, progress=False)

    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(state_dict)
    else:
        # 没文件也先让它跑起来
       raise FileNotFoundError(f"RAFT checkpoint not found at {ckpt_path}")


    model = model.to("cuda").eval()
    return model

@torch.no_grad()
def getRaftOptical(RaftModel,Image1,Image2):
    # x: [B, C, H, W] (im2)
    if not isinstance(Image1, torch.Tensor) or not isinstance(Image2, torch.Tensor):
        raise TypeError("`Image1` 与 `Image2` 必须为 torch.Tensor。")

    if Image1.dim() == 3:
        Image1 = Image1.unsqueeze(0)
    if Image2.dim() == 3:
        Image2 = Image2.unsqueeze(0)

    if Image1.shape != Image2.shape:
        raise ValueError("`Image1` 与 `Image2` 的尺寸必须一致。")

    device = next(RaftModel.parameters()).device
    img1_batch = RAFT_PREPROCESS(Image1.to(device, non_blocking=True)).contiguous()
    img2_batch = RAFT_PREPROCESS(Image2.to(device, non_blocking=True)).contiguous()

    with torch.no_grad():
        list_of_flows = RaftModel(img2_batch, img1_batch)
    predicted_flows = list_of_flows[-1]
    
    return predicted_flows


import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)

def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ

def get_sobel_kernels():
    """
    Returns Sobel kernels for gradient calculation.
    """
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return kernel_x, kernel_y


# def calculate_motion_boundary(flow, alpha=0.01, beta=0.002):
#     """
#     Calculates motion boundaries based on flow gradients as described in the paper.
#     Formula: |∇u|^2 + |∇v|^2 > alpha * (|u|^2 + |v|^2) + beta
#     """
#     flow_u = flow[:, 0:1, :, :]
#     flow_v = flow[:, 1:2, :, :]

#     sobel_x, sobel_y = get_sobel_kernels()
#     sobel_x = sobel_x.to(flow.device)
#     sobel_y = sobel_y.to(flow.device)

#     # Calculate gradients of flow components
#     grad_u_x = F.conv2d(flow_u, sobel_x, padding='same')
#     grad_u_y = F.conv2d(flow_u, sobel_y, padding='same')
#     grad_v_x = F.conv2d(flow_v, sobel_x, padding='same')
#     grad_v_y = F.conv2d(flow_v, sobel_y, padding='same')

#     grad_u_sq = grad_u_x**2 + grad_u_y**2
#     grad_v_sq = grad_v_x**2 + grad_v_y**2
    
#     motion_boundary_sum_sq_grad = grad_u_sq + grad_v_sq
    
#     mag_sq_flow = torch.sum(flow**2, dim=1, keepdim=True)

#     motion_boundary_threshold = alpha * mag_sq_flow + beta

#     motion_boundary_mask = (motion_boundary_sum_sq_grad > motion_boundary_threshold).float()
    
#     return motion_boundary_mask.squeeze(1)

def calculate_motion_boundary(flow, alpha=0.01, beta=0.002):
    """
    Calculates motion boundaries based on flow gradients as described in the paper.
    Formula: |∇u|^2 + |∇v|^2 > alpha * (|u|^2 + |v|^2) + beta
    """
    # --- 修改点 2: 在卷积前手动进行镜像填充 ---
    # Sobel 核是 3x3，所以我们需要在图像的上下左右各填充1个像素。
    # 'replicate' 模式会复制边缘像素，效果类似于镜像，且在PyTorch中实现简单。
    padding = (1, 1, 1, 1)
    padded_flow = F.pad(flow, padding, mode='replicate')
    
    flow_u = padded_flow[:, 0:1, :, :]
    flow_v = padded_flow[:, 1:2, :, :]

    sobel_x, sobel_y = get_sobel_kernels()
    sobel_x = sobel_x.to(flow.device)
    sobel_y = sobel_y.to(flow.device)

    # 在已经手动填充过的图像上进行卷积，因此这里 padding 设置为 0
    grad_u_x = F.conv2d(flow_u, sobel_x, padding=0)
    grad_u_y = F.conv2d(flow_u, sobel_y, padding=0)
    grad_v_x = F.conv2d(flow_v, sobel_x, padding=0)
    grad_v_y = F.conv2d(flow_v, sobel_y, padding=0)

    grad_u_sq = grad_u_x**2 + grad_u_y**2
    grad_v_sq = grad_v_x**2 + grad_v_y**2
    
    motion_boundary_sum_sq_grad = grad_u_sq + grad_v_sq
    
    mag_sq_flow = torch.sum(flow**2, dim=1, keepdim=True)

    motion_boundary_threshold = alpha * mag_sq_flow + beta

    motion_boundary_mask = (motion_boundary_sum_sq_grad > motion_boundary_threshold).float()
    
    return motion_boundary_mask.squeeze(1)



def forward_backward_consistency_check_with_boundary(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5,
                                       alpha_motion_boundary=0.01,
                                       beta_motion_boundary=0.002
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # Disocclusion parameters from Sundaram et al. [24], as used in the paper.
    # Motion boundary parameters from Ruder et al. (the provided paper).
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2

    # 1. Disocclusion Check (as in your original code)
    # This checks where the forward flow and warped backward flow disagree.
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    mag_fwd = torch.norm(fwd_flow, dim=1) + torch.norm(warped_bwd_flow, dim=1)
    mag_bwd = torch.norm(bwd_flow, dim=1) + torch.norm(warped_fwd_flow, dim=1)

    threshold_fwd = alpha * mag_fwd + beta
    threshold_bwd = alpha * mag_bwd + beta
    
    fwd_disocclusion_mask = (diff_fwd > threshold_fwd).float()
    bwd_disocclusion_mask = (diff_bwd > threshold_bwd).float()

    # 2. Motion Boundary Detection
    # The paper detects motion boundaries using the backward flow.
    # We will apply this logic to both forward and backward flows to get respective boundary masks.
    fwd_motion_boundary_mask = calculate_motion_boundary(fwd_flow, alpha_motion_boundary, beta_motion_boundary)
    bwd_motion_boundary_mask = calculate_motion_boundary(bwd_flow, alpha_motion_boundary, beta_motion_boundary)


    # 3. Combine Masks
    # The final temporal consistency loss excludes both disoccluded regions and motion boundaries.
    # This implies that a pixel is invalid (masked) if it is either a disocclusion OR a motion boundary.
    # A value of 1 indicates an invalid/occluded region.
    fwd_occ = torch.clamp(fwd_disocclusion_mask + fwd_motion_boundary_mask, 0, 1)
    bwd_occ = torch.clamp(bwd_disocclusion_mask + bwd_motion_boundary_mask, 0, 1)

    return fwd_occ, bwd_occ