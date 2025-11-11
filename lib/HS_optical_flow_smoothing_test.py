import torch
import torch.nn as nn
import torch.optim as optim

def horn_schunck_optimization(Ix, Iy, It, initial_flow, lambda_, num_iter=100, lr=0.01):
    """
    Horn-Schunck光流优化实现
    
    参数:
        Ix (Tensor): 图像空间梯度x分量 (1,1,H,W)
        Iy (Tensor): 图像空间梯度y分量 (1,1,H,W)
        It (Tensor): 图像时间梯度 (1,1,H,W)
        initial_flow (Tensor): 初始光流场 (1,2,H,W)
        lambda_ (float): 平滑项权重参数
        num_iter (int): 优化迭代次数
        lr (float): 学习率
        
    返回:
        Tensor: 优化后的光流场 (1,2,H,W)
    """
    # 将初始光流设置为可优化参数
    flow = nn.Parameter(initial_flow.clone().detach())
    optimizer = optim.Adam([flow], lr=lr)
    
    lambda_sq = lambda_ ** 2
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        
        u = flow[:, 0:1]  # 提取u分量 (1,1,H,W)
        v = flow[:, 1:2]  # 提取v分量 (1,1,H,W)
        
        # 数据项计算
        data_term = (Ix*u + Iy*v + It).pow(2).mean()
        
        # 平滑项计算（使用前向差分）
        # x方向梯度 (排除最后一列)
        ux = u[:, :, :, 1:] - u[:, :, :, :-1]
        vx = v[:, :, :, 1:] - v[:, :, :, :-1]
        
        # y方向梯度 (排除最后一行)
        uy = u[:, :, 1:, :] - u[:, :, :-1, :]
        vy = v[:, :, 1:, :] - v[:, :, :-1, :]
        
        # 梯度平方和平均
        smooth_term = (ux**2 + uy**2 + vx**2 + vy**2).mean()
        
        # 总损失
        loss = data_term + lambda_sq * smooth_term
        
        # 反向传播优化
        loss.backward()
        optimizer.step()
    
    return flow.data.detach()

# 示例用法
if __name__ == "__main__":
    # 假设输入参数
    H, W = 256, 256
    Ix = torch.randn(1, 1, H, W)
    Iy = torch.randn(1, 1, H, W)
    It = torch.randn(1, 1, H, W)
    initial_flow = torch.zeros(1, 2, H, W)
    
    # 优化光流
    optimized_flow = horn_schunck_optimization(
        Ix, Iy, It, 
        initial_flow,
        lambda_=0.1,
        num_iter=200,
        lr=0.05
    )
    
    print("优化后光流形状:", optimized_flow.shape)