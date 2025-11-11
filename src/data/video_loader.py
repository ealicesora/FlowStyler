"""视频数据加载模块"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torchvision.transforms as T

class VideoLoader:
    """视频帧加载器"""
    
    def __init__(self, video_dir, target_size=None, keep_aspect_ratio=True):
        """
        Args:
            video_dir: 视频帧目录
            target_size: 目标尺寸 (height, width)，如果为None则保持原始尺寸
            keep_aspect_ratio: 是否保持宽高比
        """
        self.video_dir = Path(video_dir)
        self.target_size = target_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.input_files = self._find_frames()
        self.original_size = None
    
    def _find_frames(self):
        """查找所有帧文件"""
        jpg_files = sorted(list(self.video_dir.glob('*.jpg')))
        png_files = sorted(list(self.video_dir.glob('*.png')))
        return sorted(jpg_files + png_files)
    
    def __len__(self):
        return len(self.input_files)
    
    def load_all_frames(self, device='cuda'):
        """加载所有帧"""
        frames_list = []
        print(self.video_dir)
        for file_path in self.input_files:
            frame = self._load_single_frame(file_path)
            frames_list.append(frame)
        
        frames = torch.stack(frames_list, dim=0)
        frames = self._preprocess(frames)
        
        return frames.to(device)
    
    def _load_single_frame(self, file_path):
        """加载单帧"""
        im = np.array(Image.open(str(file_path))).astype(np.float64) / 255.0
        
        # 记录原始尺寸
        if self.original_size is None:
            self.original_size = (im.shape[0], im.shape[1])
        
        if self.target_size is not None:
            target_h, target_w = self.target_size
            orig_h, orig_w = im.shape[:2]
            
            if self.keep_aspect_ratio:
                # 计算缩放比例，保持宽高比
                scale = min(target_h / orig_h, target_w / orig_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                
                # 调整尺寸
                im_resized = cv2.resize(im[:, :, :3], (new_w, new_h))
                
                # 创建目标尺寸的画布（填充黑色）
                im_padded = np.zeros((target_h, target_w, 3), dtype=np.float64)
                
                # 将调整后的图像居中放置
                start_y = (target_h - new_h) // 2
                start_x = (target_w - new_w) // 2
                im_padded[start_y:start_y+new_h, start_x:start_x+new_w] = im_resized
                
                im = im_padded
            else:
                # 直接拉伸到目标尺寸
                im = cv2.resize(im[:, :, :3], (target_w, target_h))
        
        return torch.from_numpy(im).permute(2, 0, 1).float()
    
    def _preprocess(self, frames):
        """预处理"""
        transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
        ])
        return transform(frames)
    
    def get_size_info(self):
        """获取尺寸信息"""
        info = {
            'original_size': self.original_size,
            'target_size': self.target_size,
            'resized': self.target_size is not None
        }
        return info
