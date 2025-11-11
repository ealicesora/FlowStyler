"""后处理工具函数"""
import torch
import numpy as np
from scipy.interpolate import UnivariateSpline

def temporal_brightness_smoothing(frames_tensor, smooth_factor=0.8):
    """时域亮度平滑"""
    n, c, h, w = frames_tensor.shape
    
    # 计算每帧亮度
    weights = torch.tensor([0.299, 0.587, 0.114], device=frames_tensor.device)
    brightness = (frames_tensor * weights.view(1, 3, 1, 1)).sum(dim=1)
    frame_means = brightness.mean(dim=(1,2)).cpu().numpy()
    
    # 样条平滑
    x = np.arange(n)
    spline = UnivariateSpline(x, frame_means, s=n*(1-smooth_factor)**2)
    smoothed_means = spline(x)
    
    # 调整亮度
    ratio = torch.from_numpy(smoothed_means / (frame_means + 1e-7)).float()
    ratio = ratio.view(n, 1, 1, 1).to(frames_tensor.device)
    
    return torch.clamp(frames_tensor * ratio, 0, 1)

