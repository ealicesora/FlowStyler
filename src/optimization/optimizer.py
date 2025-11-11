"""优化器模块"""
import torch
import torch.optim as optim
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from lib.ds_vectorFieldSmooth import *

class VideoStyleOptimizer:
    """视频风格迁移优化器"""
    
    def __init__(self, field_system, stylizer, config):
        """
        Args:
            field_system: LagrangianFieldSystem
            stylizer: Stylizer
            config: 配置字典
        """
        self.field_system = field_system
        self.stylizer = stylizer
        self.config = config
        
        # 创建优化器
        self.optimizer = None
        self._create_optimizer()
    
    def _create_optimizer(self):
        """创建优化器"""
        parameters = self.field_system.get_parameters()
        self.optimizer = optim.AdamW(parameters)
    
    def optimize_single_frame(self, frame, num_steps, 
                             occlusion_mask=None, step_offset=0):
        """
        优化单帧
        
        Args:
            frame: (1, 3, H, W)
            num_steps: 优化步数
            occlusion_mask: (1, 1, H, W) 遮挡掩码
            step_offset: 步数偏移（用于计算衰减）
        
        Returns:
            losses: 损失列表
        """
        losses = []
        
        for step in range(num_steps):
            # 更新粒子颜色
            with torch.no_grad():
                self.field_system.update_particle_colors(frame)
            
            # 计算损失
            print_loss = (step % 10 == 0)
            loss = self.stylizer.compute_loss(
                self.field_system.compose_image,
                frame,
                print_loss
            )
            
            # 添加正则化
            loss = loss + self._compute_regularization()
            
            # 反向传播
            loss.backward()
            
            # 梯度处理
            if occlusion_mask is not None and step > 0:
                self._apply_occlusion_mask(occlusion_mask, step + step_offset)
            
            self._smooth_gradients()
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            losses.append(loss.item())
        
        return losses
    
    def _compute_regularization(self):
        """计算正则化损失"""
        config = self.config['regularization']
        
        flow = self.field_system.warp_field.getData()
        
        smooth_loss = (
            config['lambda_l2'] * l2_regularization(flow) +
            config['lambda_tv'] * tv_regularization(flow) +
            config['lambda_huber'] * huber_regularization(flow, delta=1.0)
        ) * 1e-1
        
        # 矩阵正则化
        alpha_image = self.field_system.alpha_field.get_colored_reshaped(
            self.field_system.larg_field.downsampledShape
        )
        matrix_reg = self.field_system.larg_field.getMatrixRegLoss(
            alpha_image=alpha_image,
            alphamode=self.config['fields']['alpha_mode'],
            weight=config['matrix_reg_weight']
        )
        
        return smooth_loss + matrix_reg
    
    def _apply_occlusion_mask(self, mask, step):
        """应用遮挡感知梯度掩码"""
        from torchvision import transforms
        
        # 构建权重掩码
        step_atten = 1.02 ** step
        
        def mask_grad(field, base_weight):
            if field.output_ori.grad is None:
                return
            
            current_mask = 1.0 - mask
            reweighted_mask = (current_mask + base_weight * step_atten).clamp(0, 1)
            
            # 高斯平滑
            blur = transforms.GaussianBlur(5, 1.0)
            # reweighted_mask 已经是 [C, H, W] 格式，直接使用
            reweighted_mask = blur(reweighted_mask)
            
            # Resize到场的尺寸
            reweighted_mask = field.translateShapeMatchThisField(reweighted_mask)
            
            # 应用
            field.output_ori.grad = field.output_ori.grad * reweighted_mask
        
        mask_grad(self.field_system.alpha_field, 1e-1)
        mask_grad(self.field_system.warp_field, 1e-2)
    
    def _smooth_gradients(self):
        """平滑梯度"""
        from torchvision import transforms
        
        def smooth_grad(field, sigma=1.0, radius=5):
            if field.output_ori.grad is None:
                return
            transform = transforms.GaussianBlur(radius, sigma)
            field.output_ori.grad = transform(field.output_ori.grad)
        
        smooth_grad(self.field_system.alpha_field, sigma=1.0)
        smooth_grad(self.field_system.warp_field, sigma=1.0)
    
    def propagate_adam_momentum(self, optical_flow, mask):
        """传播Adam动量"""
        self.field_system.warp_field.UpdateAdamPara_fromOpt(
            self.optimizer, optical_flow, mask
        )
        self.field_system.alpha_field.UpdateAdamPara_fromOpt(
            self.optimizer, optical_flow, mask
        )
        
        self.field_system.warp_field.DumpPara_toOpt(self.optimizer)
        self.field_system.alpha_field.DumpPara_toOpt(self.optimizer)
