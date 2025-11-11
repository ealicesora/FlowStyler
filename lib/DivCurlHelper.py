import torch
import torch.nn.functional as F

def compute_div_curl(v):
    """
    计算给定向量场 v 的散度(div)和旋度(curl)
    其中 v 形状为 (N, 2, H, W)
    并使用复制边界 (border replicate) 的中心差分。
    
    返回:
        div: (N, H, W)
        curl: (N, H, W)
    """
    # v_x, v_y 分别是 (N,H,W)
    v_x = v[:, 0, :, :]
    v_y = v[:, 1, :, :]

    # 定义对 x 方向的中心差分
    # 先在最右和最左做 replicate padding，之后再做差分
    def partial_x(u):
        # u 形状: (N, H, W)
        # 在最后一个维度(W)左右各 padding 1
        u_pad = F.pad(u, (1, 1, 0, 0), mode='replicate')
        # 中心差分: (u(x+1) - u(x-1)) / 2
        return 0.5 * (u_pad[..., 2:] - u_pad[..., :-2])

    # 定义对 y 方向的中心差分
    # 在倒数第二个维度(H)上下各 padding 1
    def partial_y(u):
        # u 形状: (N, H, W)
        u_pad = F.pad(u, (0, 0, 1, 1), mode='replicate')
        return 0.5 * (u_pad[..., 2:, :] - u_pad[..., :-2, :])

    # 散度: div = d(v_x)/dx + d(v_y)/dy
    div = partial_x(v_x) + partial_y(v_y)
    # 旋度(在 2D 中即标量形式): curl = d(v_y)/dx - d(v_x)/dy
    curl = partial_x(v_y) - partial_y(v_x)

    return div, curl

import torch
import torch.nn.functional as F

def compute_local_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """
    计算一个 2D 向量场的局部平滑程度。
    flow: 形状为 (2, H, W)，表示每个像素点上的向量 (u,v)。
    
    返回值:
        grad_magnitude: 形状为 (H, W)，表示每个位置的梯度大小，数值越大说明该像素周围变化越剧烈，越不平滑。
    """
    # flow 的形状假设是 (2, H, W)
    # 分别计算在 x 方向和 y 方向上的离散梯度
    dx = flow[:, :, 1:] - flow[:, :, :-1]  # 结果形状 (2, H, W-1)
    dy = flow[:, 1:, :] - flow[:, :-1, :]  # 结果形状 (2, H-1, W)

    # 为了让 dx, dy 和原图对齐，我们可以对其进行 padding（补一列/行），保证输出与输入的 H, W 一致
    dx_pad = F.pad(dx, (0, 1, 0, 0))  # 在宽度方向右边补一列
    dy_pad = F.pad(dy, (0, 0, 0, 1))  # 在高度方向底部补一行

    # 由于 flow 在第 0 维度上是 (u, v)，这里先对 (u, v) 两个分量做平方和，再开平方
    # 这样得到的就是对每个像素点梯度向量的范数
    grad_magnitude = torch.sqrt((dx_pad ** 2 + dy_pad ** 2).sum(dim=0))  # 形状 (H, W)

    return grad_magnitude


# if __name__ == "__main__":
#     # 举个简单例子，随机生成一个 (2, H, W) 的向量场
#     H, W = 5, 5
#     flow = torch.rand(2, H, W)
#     smoothness_map = compute_local_smoothness(flow)

#     print("flow:", flow)
#     print("smoothness_map:", smoothness_map)
