"""风格迁移器模块"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from lib.stylizer import NNFMstylizer

class Stylizer:
    """风格迁移器包装类"""
    
    def __init__(self, style_img, content_img, config):
        """
        Args:
            style_img: (1, 3, H, W)
            content_img: (1, 3, H, W)
            config: 配置字典
        """
        self.config = config
        
        self.stylizer = NNFMstylizer(
            style_img,
            content_img,
            contentweight=config['style']['content_weight'],
            style_weight=config['style']['weight']
        )
        
        self.stylizer.enable_Random_Perspective = False
    
    def set_weights(self, *, style_weight=None, content_weight=None):
        """调整损失权重"""
        if style_weight is not None:
            self.stylizer.style_weight = style_weight
        if content_weight is not None:
            self.stylizer.content_weight = content_weight
    
    def set_random_perspective(self, enable: bool):
        """启用或关闭随机透视增强"""
        self.stylizer.enable_Random_Perspective = enable
    
    def get_inner(self):
        """返回底层 stylizer 实例，供高级控制使用"""
        return self.stylizer
    
    def compute_loss(self, compose_fn, content_img, print_loss=False):
        """
        计算风格迁移损失
        
        Args:
            compose_fn: 图像合成函数
            content_img: 当前帧
            print_loss: 是否打印
        
        Returns:
            loss: scalar tensor
        """
        return self.stylizer.IndirectForwardLoss(
            compose_fn,
            content_img,
            print_loss
        )

