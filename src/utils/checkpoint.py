"""检查点管理工具"""
import torch
from pathlib import Path
import json
from datetime import datetime

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir):
        """
        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, warp_fields, alpha_fields, metadata=None):
        """
        保存场参数
        
        Args:
            warp_fields: (N, C, H, W) tensor
            alpha_fields: (N, C, H, W) tensor
            metadata: 元数据字典
        """
        # 保存场参数
        torch.save(warp_fields.cpu(), self.checkpoint_dir / "warp_fields.pt")
        torch.save(alpha_fields.cpu(), self.checkpoint_dir / "alpha_fields.pt")
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['num_frames'] = len(warp_fields)
        metadata['warp_shape'] = list(warp_fields.shape)
        metadata['alpha_shape'] = list(alpha_fields.shape)
        
        with open(self.checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 检查点已保存到: {self.checkpoint_dir}")
    
    def load(self):
        """
        加载场参数
        
        Returns:
            warp_fields: (N, C, H, W) tensor
            alpha_fields: (N, C, H, W) tensor
            metadata: 元数据字典
        """
        warp_fields = torch.load(self.checkpoint_dir / "warp_fields.pt")
        alpha_fields = torch.load(self.checkpoint_dir / "alpha_fields.pt")
        
        metadata = {}
        metadata_file = self.checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        print(f"✅ 从 {self.checkpoint_dir} 加载检查点")
        print(f"   帧数: {metadata.get('num_frames', 'N/A')}")
        print(f"   时间: {metadata.get('timestamp', 'N/A')}")
        
        return warp_fields, alpha_fields, metadata
    
    def exists(self):
        """检查检查点是否存在"""
        return (self.checkpoint_dir / "warp_fields.pt").exists() and \
               (self.checkpoint_dir / "alpha_fields.pt").exists()

