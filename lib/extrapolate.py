import torch
import torch.nn.functional as F

def extrapolate_pytorch(mask, tensor, max_iterations=10):
    """
    使用PyTorch直接实现的外插算法，将valid区域的值传播到invalid区域。
    
    Args:
        mask (torch.Tensor): 形状为 (1, H, W)，表示有效像素的布尔掩码。
        tensor (torch.Tensor): 形状为 (C, H, W)，表示具体的值。
        max_iterations (int): 最大迭代次数。
        
    Returns:
        torch.Tensor: 外插后的tensor。
    """
    assert mask.shape[1:] == tensor.shape[1:], "Mask and tensor shapes must match on H and W dimensions."
    
    # 初始化
    mask = mask[0]  # 提取 (H, W) 维度
    result = tensor.clone()

    # 定义膨胀的卷积核
    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32, device=tensor.device).unsqueeze(0).unsqueeze(0)

    for _ in range(max_iterations):
        # 使用卷积膨胀 mask
        expanded_mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0).float(), kernel, padding=1) > 0
        expanded_mask = expanded_mask[0, 0]

        # 找到新边界 (expanded_mask 中为 True，但原 mask 中为 False)
        new_boundary = expanded_mask & ~mask

        # 对新边界进行插值
        for c in range(tensor.shape[0]):  # 遍历通道
            values = result[c].unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
            # 使用卷积计算邻域和及有效像素个数
            neighbor_sum = F.conv2d(values * mask.float(), kernel, padding=1)
            neighbor_count = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1)
            
            # 计算新边界点的平均值
            neighbor_avg = neighbor_sum / (neighbor_count + 1e-6)  # 防止除以 0
            result[c][new_boundary] = neighbor_avg[0, 0][new_boundary]

        # 更新 mask
        mask = expanded_mask

    return result


import torch
import torch.nn.functional as F

def fluid_extrapolate_8neighbors(data: torch.Tensor,
                                 mask: torch.Tensor,
                                 iterations: int = 10) -> (torch.Tensor, torch.Tensor):
    """
    使用类似流体模拟的外插思路，将 mask=0 的无效区域使用临近的有效值(8邻域)填充。
    data: (C, H, W)，浮点数
    mask: (1, H, W)，0 or 1
    iterations: 外插的迭代次数
    返回: (填充后的 data, 更新后的 mask)
    """
    device = data.device
    
    # 把 data 扩展成 (N=1, C, H, W)，mask 扩展成 (N=1, 1, H, W)
    data = data.unsqueeze(0)  # [1, C, H, W]
    mask = mask.unsqueeze(0)  # [1, 1, H, W]
    
    N, C, H, W = data.shape
    
    # 8 邻域的卷积核 (中心为 0，其余 8 个位置为 1)
    kernel_8 = torch.tensor([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)  # (out_ch, in_ch, kH, kW)

    kernel_8 = kernel_8.repeat(C, 1, 1, 1)
    
    for _ in range(iterations):
        # 对 mask 做卷积，得到每个像素周围有效像素(8邻域)的数量
        valid_count = F.conv2d(mask, kernel_8, padding=1)  # [1, 1, H, W]

        # 对 data 做同样的卷积，得到周围像素值加和
        neighbor_sum = F.conv2d(data, kernel_8, padding=1,groups=C)  # [1, C, H, W]

        # 计算平均值，避免除以 0 在分母加上一个极小值
        avg_val = neighbor_sum / (valid_count + 1e-8)

        # 找到哪些像素目前是无效(mask=0)，但有至少一个有效邻居(valid_count>0)
        to_fill = (mask < 0.5) & (valid_count > 0)

        # 用平均值去填充无效像素
        data = torch.where(to_fill, avg_val, data)

        # 更新 mask，将这些填充过的像素标记为有效
        mask = torch.where(to_fill, torch.ones_like(mask), mask)

    # 去掉批次维度
    data = data.squeeze(0)
    mask = mask.squeeze(0)

    return data#, mask


def erode_mask_withBlackBorder(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    对输入 mask (形状: [1, H, W]) 进行形态学腐蚀 (binary erosion)。
    - 使用 3x3 全 1 的结构元素
    - 迭代多次可以让腐蚀范围继续扩大
    
    参数:
    mask: (1, H, W), 二值 (0/1)，表示 valid/invalid
    iterations: 腐蚀的迭代次数
    
    返回:
    eroded_mask: (1, H, W), 二值 (0/1)，腐蚀后的 mask
    """
    # 先扩展到 (N=1, C=1, H, W) 以便使用 conv2d
    mask_4d = mask.unsqueeze(0)  # [1, 1, H, W]
    
    # 3x3 全 1 的卷积核(结构元素)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)
    
    # 迭代腐蚀
    for _ in range(iterations):
        # 对 mask 做卷积，统计 3x3 区域内的和
        # 若在 3x3 区域中全是 1，则和为 9(因为中心+周围共9个像素)
        neighbor_sum = F.conv2d(mask_4d, kernel, padding=1)  # [1, 1, H, W]
        
        # 只有当 3x3 全部为 1 时，neighbor_sum 才等于 9
        # 因此我们判断 neighbor_sum == 9 来保留 1，否则变成 0
        mask_4d = (neighbor_sum > 8.9).float()
    
    # 把 batch 维度去掉
    eroded_mask = mask_4d.squeeze(0)  # 回到 (1, H, W)
    return eroded_mask


import torch
import torch.nn.functional as F

def erode_mask_new(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    对输入 mask (形状: [1, H, W]) 进行形态学腐蚀 (binary erosion)。
    - 使用 3x3 全 1 的结构元素
    - 迭代多次可以让腐蚀范围继续扩大
    
    参数:
    mask: (1, H, W), 二值 (0/1)，表示 valid/invalid
    iterations: 腐蚀的迭代次数
    
    返回:
    eroded_mask: (1, H, W), 二值 (0/1)，腐蚀后的 mask
    """
    # 先扩展到 (N=1, C=1, H, W) 以便使用 conv2d
    mask_4d = mask.unsqueeze(0)  # [1, 1, H, W]
    
    # 3x3 全 1 的卷积核(结构元素)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)
    
    # 手动扩展边界以模拟 reflect padding
    for _ in range(iterations):
        mask_4d = F.pad(mask_4d, (1, 1, 1, 1), mode='reflect')  # 边界扩展
        neighbor_sum = F.conv2d(mask_4d, kernel)  # [1, 1, H, W]
        
        # 只有当 3x3 全部为 1 时，neighbor_sum 才等于 9
        mask_4d = (neighbor_sum == 9).float()
    
    # 把 batch 维度去掉
    eroded_mask = mask_4d.squeeze(0)  # 回到 (1, H, W)
    return eroded_mask


import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F

def fluid_extrapolate_8neighbors_multichannel(
    data: torch.Tensor, 
    mask: torch.Tensor, 
    iterations: int = 10
) -> (torch.Tensor, torch.Tensor):
    """
    使用类似流体模拟的外插思路，将 mask=0 的无效区域使用临近的有效值(8邻域)填充。
    支持 data 有 C 个通道，每个通道在外插时独立计算邻域平均值。

    参数:
        data: (C, H, W)，浮点数
        mask: (1, H, W)，二值(0/1)，表示无效/有效
        iterations: 外插的迭代次数

    返回:
        (data_filled, mask_filled):
            data_filled: (C, H, W)，外插填充完后的 data
            mask_filled: (1, H, W)，外插后更新的 mask
    """
    device = data.device
    
    # ----------------------
    # 1) 扩展维度以适配 conv2d
    # ----------------------
    # data: (C, H, W) -> (N=1, C, H, W)
    data = data.unsqueeze(0)  # [1, C, H, W]
    # mask: (1, H, W) -> (N=1, 1, H, W)
    mask = mask.unsqueeze(0)  # [1, 1, H, W]

    # ----------------------
    # 2) 准备 8 邻域卷积核
    #    - 用于 mask 的卷积 (单通道)
    #    - 用于 data 的卷积 (C 通道，group 卷积)
    # ----------------------
    # 8 邻域, 中心=0, 其余8个=1
    kernel_8_single = torch.tensor([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # 用于 mask 的卷积: [out_ch=1, in_ch=1, kH=3, kW=3]
    kernel_mask = kernel_8_single  # 直接单通道

    # 用于 data 的卷积: 因为 data 有 C 通道，希望对每个通道独立做同样的卷积
    # => 使用 group 卷积: groups=C
    # => kernel shape: [C, 1, 3, 3]
    C = data.shape[1]
    kernel_data = kernel_8_single.repeat(C, 1, 1, 1)  # [C, 1, 3, 3]

    # ----------------------
    # 3) 循环迭代外插
    # ----------------------
    for _ in range(iterations):
        # (a) 计算 mask 的邻域有效像素数量 (valid_count)
        # mask.shape = [1, 1, H, W]
        # kernel_mask.shape = [1, 1, 3, 3]
        valid_count = F.conv2d(mask, kernel_mask, padding=1)  # [1, 1, H, W]

        # (b) 计算 data 的邻域像素值加和 (neighbor_sum)
        # data.shape = [1, C, H, W]
        # kernel_data.shape = [C, 1, 3, 3]
        # groups = C => 对每个通道独立卷积
        neighbor_sum = F.conv2d(data, kernel_data, padding=1, groups=C)  # [1, C, H, W]

        # (c) 计算平均值
        # 避免除以 0，在分母加 1e-8
        avg_val = neighbor_sum / (valid_count + 1e-8)  # [1, C, H, W]

        # (d) 找到需要外插的位置
        # mask=0(无效) 且 valid_count>0(至少有1个有效邻居)
        to_fill = (mask < 0.5) & (valid_count > 0)  # [1, 1, H, W]

        # (e) 用平均值填充
        # to_fill 会在 [1,1,H,W]，而 data 是 [1,C,H,W]
        # PyTorch 会自动 broadcast (1,1,H,W) -> (1,C,H,W)
        data = torch.where(to_fill, avg_val, data)

        # (f) 更新 mask: 这些被填充的位置，标记为有效
        mask = torch.where(to_fill, torch.ones_like(mask), mask)

    # ----------------------
    # 4) 去掉批量维度
    # ----------------------
    data_filled = data.squeeze(0)  # [C, H, W]
    mask_filled = mask.squeeze(0)  # [1, H, W]
    
    return data_filled#, mask_filled




