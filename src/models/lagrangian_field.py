"""Lagrangian场系统模块"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from lib.Larg_field import Larg_Image
from lib.scaled_2D import Field2D

class LagrangianFieldSystem:
    """Lagrangian场系统"""
    
    def __init__(self, content_image, config):
        """
        Args:
            content_image: (1, 3, H, W)
            config: 配置字典
        """
        self.config = config
        self.target_size = content_image.size()
        
        # 创建warping场
        self.warp_field = Field2D(
            targetSize=self.target_size,
            downsample_ratio=config['fields']['downsample_ratio'],
            init_method="uv",
            channelcount=2,
            enable_DownSample_beforeUse=False
        )
        
        # 创建alpha场
        alpha_mode = config['fields']['alpha_mode']
        channelcount = 9 if alpha_mode == 'Matrix33' else 16
        

        if alpha_mode == 'alphamode':
            channelcount = 1
        elif alpha_mode == 'Matrix33':
            channelcount = 9
        elif alpha_mode == 'Simple_adding':
            channelcount = 1
        elif alpha_mode == 'Matrix44':
            channelcount = 16
        else:
            raise ValueError(f"Unsupported alpha mode: {alpha_mode}")

            
        self.alpha_field = Field2D(
            targetSize=self.target_size,
            downsample_ratio=config['fields']['downsample_ratio'],
            init_method="zeros",
            channelcount=channelcount,
            initValue=torch.tensor([0.0] * channelcount),
            enable_DownSample_beforeUse=False
        )
        
        # 创建粒子系统
        self.larg_field = Larg_Image(
            content_image,
            downSample_ratio=config['fields']['downsample_ratio']
        )
        
        # 设置学习率缩放
        self.warp_field.lr_scale = config['fields']['warp_lr_scale']
        self.alpha_field.lr_scale = config['fields']['alpha_lr_scale']
    
    def compose_image(self):
        """从场参数合成输出图像"""
        # 更新粒子位置
        updated_pos = self.larg_field.getUpdatedParticlePostionFromWarpingField(
            self.warp_field.getData()
        )
        
        # 获取alpha场
        alpha_image = self.alpha_field.get_colored_reshaped(
            self.larg_field.downsampledShape
        )
        
        # 渲染
        output = self.larg_field.getImage_byNewposition(
            updated_pos,
            kernel_size=2,
            effectRangeSize=2,
            alpha_image=alpha_image,
            alphamode=self.config['fields']['alpha_mode']
        )
        
        return output
    
    def get_parameters(self):
        """获取可优化参数"""
        lr = self.config['optimization']['lr']
        return [
            {'params': self.warp_field.get_parameter()[0],
             'lr': lr * self.warp_field.lr_scale},
            {'params': self.alpha_field.get_parameter()[0],
             'lr': lr * self.alpha_field.lr_scale}
        ]
    
    def propagate_with_optical_flow(self, optical_flow):
        """使用光流传播场"""
        with torch.no_grad():
            self.warp_field.advect_withOptical_andUpdate(optical_flow)
            self.alpha_field.advect_withOptical_andUpdate(optical_flow)
        
    def update_particle_colors(self, frame):
        """更新粒子颜色"""
        self.larg_field.setNewImage(frame)
