import torch
import torch.nn.functional as F

def anisotropic_diffusion_torch(vector_field, iterations=10, k=0.1, delta_t=0.1, device="cuda"):
    """
    使用 PyTorch 实现的各向异性扩散平滑，支持 CUDA 加速
    :param vector_field: 输入的向量场 (1, 2, h, w)
    :param iterations: 扩散迭代次数
    :param k: 边缘保留参数
    :param delta_t: 时间步长
    :param device: 设备，默认为 CUDA
    :return: 平滑后的向量场 (1, 2, h, w)
    """
    # 将向量场转移到指定设备
    vector_field = vector_field.to(device)
    U_x = vector_field[:, 0, :, :].clone()  # x 分量
    U_y = vector_field[:, 1, :, :].clone()  # y 分量

    # 定义 Sobel 卷积核
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    for _ in range(iterations):
        # 计算梯度 (x 和 y 分量的 Sobel 滤波)
        grad_Ux_x = F.conv2d(U_x.unsqueeze(1), sobel_x, padding=1).squeeze(1)  # x 分量 x 方向梯度
        grad_Ux_y = F.conv2d(U_x.unsqueeze(1), sobel_y, padding=1).squeeze(1)  # x 分量 y 方向梯度
        grad_Uy_x = F.conv2d(U_y.unsqueeze(1), sobel_x, padding=1).squeeze(1)  # y 分量 x 方向梯度
        grad_Uy_y = F.conv2d(U_y.unsqueeze(1), sobel_y, padding=1).squeeze(1)  # y 分量 y 方向梯度

        # 梯度幅值（模）
        grad_magnitude = torch.sqrt(grad_Ux_x**2 + grad_Ux_y**2 + grad_Uy_x**2 + grad_Uy_y**2)

        # 计算扩散系数 c(s)
        c = torch.exp(-(grad_magnitude / k)**2)  # Perona-Malik 方案 1

        # 计算扩散量（导数项）
        divergence_Ux = F.conv2d((c * grad_Ux_x).unsqueeze(1), sobel_x, padding=1).squeeze(1) + \
                        F.conv2d((c * grad_Ux_y).unsqueeze(1), sobel_y, padding=1).squeeze(1)
        divergence_Uy = F.conv2d((c * grad_Uy_x).unsqueeze(1), sobel_x, padding=1).squeeze(1) + \
                        F.conv2d((c * grad_Uy_y).unsqueeze(1), sobel_y, padding=1).squeeze(1)

        # 更新向量场
        U_x = U_x + delta_t * divergence_Ux
        U_y = U_y + delta_t * divergence_Uy

    # 重组为向量场
    smoothed_vector_field = torch.stack([U_x, U_y], dim=1).unsqueeze(0)
    return smoothed_vector_field

# # 示例调用
# h, w = 100, 100
# vector_field = torch.rand(1, 2, h, w)  # 随机生成一个向量场
# smoothed_field = anisotropic_diffusion_torch(vector_field, iterations=20, k=0.2, delta_t=0.1, device="cuda")

# # 将结果转回 CPU 并转换为 numpy 以便可视化或后续处理
# smoothed_field_cpu = smoothed_field.cpu().numpy()
# print(smoothed_field_cpu.shape)  # 输出形状


def anisotropic_diffusion_scalar_torch(scalar_field, iterations=10, k=0.1, delta_t=0.1, device="cuda"):
    """
    使用 PyTorch 实现对标量场的各向异性扩散，支持 CUDA 加速
    :param scalar_field: 输入的标量场 (1, 1, h, w)
    :param iterations: 扩散迭代次数
    :param k: 边缘保留参数
    :param delta_t: 时间步长
    :param device: 设备，默认为 CUDA
    :return: 平滑后的标量场 (1, 1, h, w)
    """
    # 将标量场移动到指定设备
    scalar_field = scalar_field.to(device)

    # 定义 Sobel 卷积核
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    for _ in range(iterations):
        # 计算梯度
        grad_x = F.conv2d(scalar_field, sobel_x, padding=1)  # x 方向梯度
        grad_y = F.conv2d(scalar_field, sobel_y, padding=1)  # y 方向梯度

        # 梯度幅值（模）
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

        # 计算扩散系数 c(s)
        c = torch.exp(-(grad_magnitude / k)**2)  # Perona-Malik 方案 1

        # 计算扩散量（导数项）
        divergence_x = F.conv2d(c * grad_x, sobel_x, padding=1)
        divergence_y = F.conv2d(c * grad_y, sobel_y, padding=1)

        # 更新标量场
        scalar_field = scalar_field + delta_t * (divergence_x + divergence_y)

    return scalar_field

# # 示例调用
# h, w = 100, 100
# scalar_field = torch.rand(1, 1, h, w)  # 随机生成一个标量场
# smoothed_scalar_field = anisotropic_diffusion_scalar_torch(scalar_field, iterations=20, k=0.2, delta_t=0.1, device="cuda")

# # 将结果转回 CPU 并转换为 numpy 以便可视化或后续处理
# smoothed_scalar_field_cpu = smoothed_scalar_field.cpu().numpy()
# print(smoothed_scalar_field_cpu.shape)  # 输出形状
