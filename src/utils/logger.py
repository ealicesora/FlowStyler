"""日志工具"""
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(log_dir, name="v28_training"):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        name: 日志名称
    
    Returns:
        logger: logging.Logger对象
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建文件handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志初始化完成，日志文件: {log_file}")
    
    return logger


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir):
        self.logger = setup_logger(log_dir)
        self.losses = []
    
    def log_config(self, config):
        """记录配置"""
        self.logger.info("=" * 60)
        self.logger.info("训练配置:")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)
    
    def log_frame(self, epoch, frame_idx, total_frames, loss, direction="forward"):
        """记录帧优化信息"""
        self.logger.info(
            f"Epoch {epoch} [{direction}] - "
            f"Frame {frame_idx+1}/{total_frames} - "
            f"Loss: {loss:.4f}"
        )
        self.losses.append(loss)
    
    def log_epoch(self, epoch, avg_loss):
        """记录epoch信息"""
        self.logger.info("=" * 60)
        self.logger.info(f"Epoch {epoch} 完成 - 平均Loss: {avg_loss:.4f}")
        self.logger.info("=" * 60)
    
    def log_completion(self, total_time):
        """记录训练完成"""
        self.logger.info("=" * 60)
        self.logger.info(f"训练完成！总耗时: {total_time:.2f} 秒")
        self.logger.info("=" * 60)

