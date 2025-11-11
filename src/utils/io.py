"""IO工具函数"""
import torch
from PIL import Image
import os
import cv2
import numpy as np

def save_tensor_as_images(tensor, output_dir, filename_prefix="frame"):
    """保存tensor为图像序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    tensor = tensor.detach().cpu()
    if tensor.max() <= 1.0:
        tensor = tensor.mul(255).clamp(0, 255)
    
    images = tensor.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    for idx in range(images.shape[0]):
        filename = f"{filename_prefix}_{idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(images[idx]).save(filepath)
    
    print(f"✅ saved {images.shape[0]} frames to {output_dir}")

def save_tensor_as_video(tensor, output_path, fps=30):
    """
    保存tensor为视频文件
    
    Args:
        tensor: (N, C, H, W) 视频帧tensor
        output_path: 输出视频路径
        fps: 帧率
    """
    tensor = tensor.detach().cpu()
    if tensor.max() <= 1.0:
        tensor = tensor.mul(255).clamp(0, 255)
    
    # 转换为 numpy: (N, H, W, C)
    frames = tensor.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    n_frames, height, width, channels = frames.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(n_frames):
        # OpenCV 使用 BGR 格式
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ video saved to: {output_path}")
