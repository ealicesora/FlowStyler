import torch
import torch.nn.functional as F

import torch

def l2_regularization(flow):
    """
    Args:
        flow: 光流场，形状为 [1, 2, h, w]
    Returns:
        L2正则化损失（标量）
    """
    # 计算水平和垂直梯度 (dx, dy)
    gradients = torch.gradient(flow, dim=(2, 3))  # 对h和w维度求导
    dx = gradients[1]  # 宽度方向梯度 [1,2,h,w]
    dy = gradients[0]  # 高度方向梯度 [1,2,h,w]
    
    # 计算L2正则化项：梯度平方和
    l2_loss = torch.sum(dx**2 + dy**2)
    return l2_loss


def jacobian_regularization(flow, threshold=0.1, penalty_type='l2'):
    """
    雅可比正则化函数，用于防止位移场出现折叠
    
    Args:
        flow: 位移场，形状为 [1, 2, h, w] (第一个通道为dx，第二个通道为dy)
        threshold: 雅可比行列式的最小安全阈值，低于此值将受到惩罚
        penalty_type: 惩罚类型 ('l2', 'log', 'huber')
    
    Returns:
        雅可比正则化损失（标量）
    """
    # 获取位移场尺寸
    _, _, h, w = flow.shape
    
    # 计算位移场的空间梯度
    gradients_x = torch.gradient(flow, dim=3)  # 宽度方向梯度 (x方向)
    gradients_y = torch.gradient(flow, dim=2)  # 高度方向梯度 (y方向)
    
    # 提取各分量梯度
    dx_dx = gradients_x[0][:, 0, :, :]  # ∂(dx)/∂x
    dx_dy = gradients_y[0][:, 0, :, :]  # ∂(dx)/∂y
    dy_dx = gradients_x[0][:, 1, :, :]  # ∂(dy)/∂x
    dy_dy = gradients_y[0][:, 1, :, :]  # ∂(dy)/∂y
    
    # 计算雅可比行列式
    # J = | 1 + ∂dx/∂x   ∂dx/∂y  |
    #     |   ∂dy/∂x   1 + ∂dy/∂y |
    detJ = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx
    
    # 根据惩罚类型计算正则化损失
    if penalty_type == 'l2':
        # L2惩罚：对低于阈值的行列式进行平方惩罚
        penalty = torch.where(detJ < threshold, 
                             (threshold - detJ)**2, 
                             torch.zeros_like(detJ))
    elif penalty_type == 'log':
        # 对数障碍函数：强制行列式大于0
        penalty = -torch.log(detJ.clamp(min=1e-6))
    elif penalty_type == 'huber':
        # Huber惩罚：对严重折叠区域进行线性惩罚
        linear_region = (detJ < threshold)
        quadratic = 0.5 * (threshold - detJ)**2
        linear = threshold * (threshold - detJ) - 0.5 * threshold**2
        penalty = torch.where(linear_region, quadratic, linear)
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")
    
    # 返回平均损失（可选：对折叠区域加权）
    return penalty.mean()


def jacobian_regularization_advanced(flow, threshold=0.1, penalty_type='l2', padding_mode='replicate'):
    """
    flow: [B, 2, H, W]
    """
    B, C, H, W = flow.shape

    # 1) pad 一圈
    flow_padded = F.pad(flow, (1, 1, 1, 1), mode=padding_mode)  # [B, 2, H+2, W+2]

    # 2) 中心差分
    # x 方向差分：在最后一维上做
    dx = (flow_padded[:, :, :, 2:] - flow_padded[:, :, :, :-2]) / 2.0   # [B, 2, H+2, W]
    # y 方向差分：在倒数第二维上做
    dy = (flow_padded[:, :, 2:, :] - flow_padded[:, :, :-2, :]) / 2.0   # [B, 2, H, W+2]

    # 3) 把多出来的一圈裁掉，让它们都变回 [B, 2, H, W]
    dx = dx[:, :, 1:-1, :]      # H+2 -> H
    dy = dy[:, :, :, 1:-1]      # W+2 -> W

    # 4) 拆成 4 个偏导，尺寸现在都是 [B, H, W]
    dx_dx = dx[:, 0, :, :]   # ∂(dx)/∂x
    dx_dy = dy[:, 0, :, :]   # ∂(dx)/∂y
    dy_dx = dx[:, 1, :, :]   # ∂(dy)/∂x
    dy_dy = dy[:, 1, :, :]   # ∂(dy)/∂y

    # 5) 计算 detJ
    detJ = (1 + dx_dx) * (1 + dy_dy) - dx_dy * dy_dx  # [B, H, W]

    # 6) 惩罚
    if penalty_type == 'l2':
        penalty = torch.where(detJ < threshold,
                              (threshold - detJ) ** 2,
                              torch.zeros_like(detJ))
    elif penalty_type == 'log':
        penalty = -torch.log(detJ.clamp(min=1e-6))
    elif penalty_type == 'huber':
        linear_region = (detJ < threshold)
        quadratic = 0.5 * (threshold - detJ) ** 2
        linear = threshold * (threshold - detJ) - 0.5 * threshold ** 2
        penalty = torch.where(linear_region, quadratic, linear)
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")

    return penalty.mean()

def tv_regularization(flow):
    """
    Args:
        flow: 光流场，形状为 [1, 2, h, w]
    Returns:
        TV正则化损失（标量）
    """
    # 计算水平和垂直梯度 (dx, dy)
    gradients = torch.gradient(flow, dim=(2, 3))
    dx = gradients[1]  # [1,2,h,w]
    dy = gradients[0]  # [1,2,h,w]
    
    # 计算TV正则化项：梯度绝对值之和
    tv_loss = torch.sum(torch.abs(dx) + torch.abs(dy))
    return tv_loss


def huber_regularization(flow, delta=1.0):
    """
    Args:
        flow: 光流场，形状为 [1, 2, h, w]
        delta: Huber损失的分段阈值，默认为1.0
    Returns:
        Huber正则化损失（标量）
    """
    # 计算水平和垂直梯度 (dx, dy)
    gradients = torch.gradient(flow, dim=(2, 3))
    dx = gradients[1]  # [1,2,h,w]
    dy = gradients[0]  # [1,2,h,w]
    
    # 计算梯度幅值
    grad_magnitude = torch.sqrt(dx**2 + dy**2 + 1e-8)  # 防止除零
    
    # Huber分段函数
    huber_mask = (grad_magnitude < delta).float()
    huber_loss = (
        huber_mask * 0.5 * grad_magnitude**2 +
        (1 - huber_mask) * delta * (grad_magnitude - 0.5 * delta)
    )
    return torch.sum(huber_loss)


def edge_zero_constraint(tensor, 
                        border_size=2, 
                        constraint_strength=1.0,
                        epsilon=1e-6):
    """
    四边零值约束函数
    
    参数：
        tensor: 输入张量 [B, C, H, W]
        border_size: 边界宽度（单位：像素）
        constraint_strength: 约束强度系数
        epsilon: 数值稳定项
    
    返回：
        loss: 约束损失值
        mask: 使用的边界掩码（可视化用）
    """
    B, C, H, W = tensor.shape
    
    # 创建全1掩码（后续裁剪边界区域）
    mask = torch.ones_like(tensor)
    
    # 水平方向裁剪（排除上下边界）
    if H > 2*border_size:
        mask[..., border_size:-border_size, :] = 0
    
    # 垂直方向裁剪（排除左右边界）
    if W > 2*border_size:
        mask[..., :, border_size:-border_size] = 0
    
    # 计算约束损失（带平滑的L1损失）
    edge_values = tensor * mask
    loss = constraint_strength * torch.mean(
        torch.sqrt(edge_values**2 + epsilon)
    )
    
    return loss
