"""光流计算模块"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
import lib.Optical_Helper as OpticalHelper

class OpticalFlowComputer:
    """光流计算器"""
    
    def __init__(self, config):
        self.config = config
        self.model = OpticalHelper.GetRaftModel().cuda()
    
    def compute_all_flows(self, frames, masks=None):
        """
        计算所有帧的光流和遮挡掩码
        
        Args:
            frames: (N, 3, H, W)
            masks: optional SAM masks
        
        Returns:
            optical_fwd: (N-1, 2, H, W) 前向光流
            optical_bwd: (N, 2, H, W) 后向光流（padding）
            masks_fwd: (N-1, 1, H, W) 前向遮挡掩码
            masks_bwd: (N, 1, H, W) 后向遮挡掩码
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError("`frames` 需要是 torch.Tensor 类型。")
        if frames.dim() != 4:
            raise ValueError("`frames` 需要是形状为 (N, 3, H, W) 的张量。")

        N = frames.shape[0]
        if N < 2:
            raise ValueError("至少需要两帧图像才能计算光流。")

        H, W = frames.shape[2:]
        model_device = next(self.model.parameters()).device
        frames = frames.to(model_device, non_blocking=True)

        optical_flow_cfg = self.config.get('optical_flow', {})
        batch_size = None
        if isinstance(optical_flow_cfg, dict):
            batch_size = optical_flow_cfg.get('batch_size', None)
        else:
            batch_size = getattr(optical_flow_cfg, 'batch_size', None)

        if batch_size is None or batch_size <= 0:
            batch_size = min(32, N - 1)
        batch_size = max(1, min(batch_size, N - 1))

        # 批量构造前向与后向帧对
        frames_curr = frames[:-1].contiguous()
        frames_next = frames[1:].contiguous()

        flows_fwd = []
        flows_bwd = []
        with torch.no_grad():
            for start in range(0, N - 1, batch_size):
                end = min(start + batch_size, N - 1)
                curr = frames_curr[start:end].contiguous()
                nxt = frames_next[start:end].contiguous()

                flows_fwd.append(
                    OpticalHelper.getRaftOptical(
                        self.model,
                        curr,
                        nxt,
                    )
                )
                flows_bwd.append(
                    OpticalHelper.getRaftOptical(
                        self.model,
                        nxt,
                        curr,
                    )
                )

        optical_fwd = torch.cat(flows_fwd, dim=0)
        optical_bwd = torch.cat(flows_bwd, dim=0)

        masks_fwd_occ, masks_bwd_occ = OpticalHelper.forward_backward_consistency_check(
            optical_fwd,
            optical_bwd,
            alpha=self.config['optical_flow']['consistency_alpha'],
            beta=self.config['optical_flow']['consistency_beta']
        )

        dtype = optical_fwd.dtype
        masks_fwd = (1.0 - masks_fwd_occ.unsqueeze(1)).to(dtype=dtype)
        masks_bwd = torch.zeros((N, 1, H, W), device=model_device, dtype=dtype)
        masks_bwd[1:] = 1.0 - masks_bwd_occ.unsqueeze(1)
        
        # Padding后向光流
        optical_bwd = F.pad(optical_bwd, (0, 0, 0, 0, 0, 0, 1, 0))
        
        return optical_fwd, optical_bwd, masks_fwd, masks_bwd
    
    def process_masks(self, masks_fwd, masks_bwd, iterations=1):
        """处理遮挡掩码（腐蚀）"""
        from lib.extrapolate import erode_mask_withBlackBorder
        
        with torch.no_grad():
            # 腐蚀前向掩码
            for i in range(len(masks_fwd)):
                masks_fwd[i] = erode_mask_withBlackBorder(
                    masks_fwd[i] * 1.0, iterations=iterations
                )
            
            # 二值化
            masks_fwd = masks_fwd > 0.9
            masks_fwd = masks_fwd * 1.0
            
            # 腐蚀后向掩码
            for i in range(len(masks_bwd)):
                masks_bwd[i] = erode_mask_withBlackBorder(
                    masks_bwd[i] * 1.0, iterations=iterations
                )
            
            masks_bwd = masks_bwd > 0.9
            masks_bwd = masks_bwd * 1.0
        
        return masks_fwd, masks_bwd
