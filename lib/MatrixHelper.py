
import torch
def build_scaled_spd_matrix(params):
    """
    构建带独立缩放因子的对称正定矩阵 (H,W,3,3)
    
    参数结构:
        params: (H,W,9)
            - params[..., 0:6]: Cholesky分解参数（同原始6参数版）
            - params[..., 6:9]: 通道缩放因子（log域参数，需指数激活）
        
    返回:
        M_total: (H,W,3,3) 完整变换矩阵 = S @ M，其中：
            - M: 由Cholesky参数构建的SPD矩阵
            - S: 对角缩放矩阵（diag(exp(scale_params))）
    """
    pointsize = params.shape[-2]
    
    # 分离参数
    cholesky_params = params[..., 0:6]
    scale_params = params[..., 6:9]
    
    # --- 构建基础SPD矩阵M ---
    # 原始Cholesky参数初始化（确保M初始为I）
    L = torch.zeros(pointsize, 3, 3, device=params.device)
    L[..., 0, 0] = 1.0  # l11初始为1
    L[..., 1, 1] = 1.0  # l22初始为1
    L[..., 2, 2] = 1.0  # l33初始为1
    
    # 使用参数更新非初始化的位置（初始为0）
    L[..., 1, 0] = cholesky_params[..., 1]  # l21
    L[..., 2, 0] = cholesky_params[..., 3]  # l31
    L[..., 2, 1] = cholesky_params[..., 4]  # l32 
    
    # Softplus确保对角正定（初始强制为1）
    diag_indices = [0, 2, 5]  # 对应原始参数的l11, l22, l33位置
    for i, idx in enumerate(diag_indices):
        L[..., i, i] = torch.nn.functional.softplus(cholesky_params[..., idx]) + 1e-6
    
    M = L @ L.transpose(-1, -2)
    
    # --- 构建缩放矩阵S ---
    S = torch.zeros(pointsize, 3, 3, device=params.device)
    diag_scale = torch.exp(scale_params)  # 指数激活保证正值
    S[..., 0, 0] = diag_scale[..., 0]
    S[..., 1, 1] = diag_scale[..., 1]
    S[..., 2, 2] = diag_scale[..., 2]
    
    return S @ M  # 最终变换矩阵




def fully_diff_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """
    完全可微的旋转矩阵校正方法
    输入: 任意3x3矩阵 (..., 3, 3)
    输出: 最近的正交旋转矩阵 (行列式=1)
    """
    # SVD分解（PyTorch >=1.9支持可微SVD）
    U, S, V = torch.linalg.svd(matrix)
    
    # 计算行列式符号（通过U@V^T的行列式自身等价于符号）
    det = torch.det(U @ V.mT)  # 形状(...)，值只能是±1
    
    # 构建可微的符号修正矩阵（关键改进点）
    diag = torch.stack([torch.ones_like(det)]*2 + [det], dim=-1)  # [1,1,det]
    D = torch.diag_embed(diag)
    
    # 计算最终旋转矩阵
    return U @ D @ V.mT

