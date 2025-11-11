
import torch

def apply_mask(tensor, mask):
    """
    将 Mask 应用到梯度张量上（自动对齐维度）
    Args:
        tensor: 输入张量（如梯度 dx/dy），形状为 [B, C, H, W]
        mask: 二值掩码，形状为 [1, 1, H, W]，1 表示有效区域
    Returns:
        掩码后的张量
    """
    # 扩展 mask 到与 tensor 相同的通道数（假设 C=2，光流场 u/v）
    mask_expanded = mask.expand_as(tensor)  # [1,1,H,W] → [B,C,H,W]
    return tensor * mask_expanded

def l2_regularization(flow, mask, reduction='mean'):
    gradients = torch.gradient(flow, dim=(2, 3))
    dx, dy = gradients[1], gradients[0]  # dx: 宽度梯度, dy: 高度梯度
    dx_masked = apply_mask(dx, mask)
    dy_masked = apply_mask(dy, mask)
    l2_loss = dx_masked**2 + dy_masked**2
    return _reduce(l2_loss, reduction)

def tv_regularization(flow, mask, isotropic=True, reduction='mean'):
    gradients = torch.gradient(flow, dim=(2, 3))
    dx, dy = gradients[1], gradients[0]
    dx_masked = apply_mask(dx, mask)
    dy_masked = apply_mask(dy, mask)
    if isotropic:
        tv = torch.sqrt(dx_masked**2 + dy_masked**2 + 1e-8)
    else:
        tv = torch.abs(dx_masked) + torch.abs(dy_masked)
    return _reduce(tv, reduction)

def huber_regularization(flow, mask, delta=1.0, reduction='mean'):
    gradients = torch.gradient(flow, dim=(2, 3))
    dx, dy = gradients[1], gradients[0]
    dx_masked = apply_mask(dx, mask)
    dy_masked = apply_mask(dy, mask)
    grad_magnitude = torch.sqrt(dx_masked**2 + dy_masked**2 + 1e-8)
    huber_mask = (grad_magnitude < delta).float()
    huber_loss = (
        huber_mask * 0.5 * grad_magnitude**2 +
        (1 - huber_mask) * delta * (grad_magnitude - 0.5 * delta)
    )
    return _reduce(huber_loss, reduction)

def _reduce(tensor, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(tensor)
    elif reduction == 'sum':
        return torch.sum(tensor)
    else:
        raise ValueError("Invalid reduction. Choose 'mean' or 'sum'.")