import torch
import torch.nn.functional as F

def piecewise_smooth_in01(x: torch.Tensor,
                          k: float = 8.0,
                          shift: float = 4.0) -> torch.Tensor:
    """
    一个在 [0,1] 区间上严格等于 y=x 的分段函数：
      - 当 x < 0 时，用 logistic(...) 逼近 0，但不等于 0
      - 当 0 <= x <= 1 时，f(x)= x
      - 当 x > 1 时，用 logistic(...) 逼近 1，但不等于 1

    参数：
      x: 任何形状的张量
      k: 控制 logistic 斜率，越大，左/右两端过渡越迅速
      shift: 控制在 x=0 和 x=1 附近时，logistic 的偏移量
             使得 f(0) 时大约 ~ 1e-N，很接近 0
             使得 f(1) 时大约 ~ 1 - 1e-N，很接近 1

    返回：
      与 x 同形状的张量，取值大部分在 [0,1]，只在极小的跳变点上有不连续：
        - x=0 处：左段 logistic(...) 和中段 0 的切换
        - x=1 处：中段 1 和右段 logistic(...) 的切换
      这两个跳变非常小，可在数值层面上近似忽略。
    """
    # 这里定义一个简单的 logistic 函数
    def logistic(t: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-t))
    
    # ------- 左段：x<0 -------
    # 让它随着 x-> -∞ 时非常接近 0
    # 当 x=0 时 logistic(k*(0 - shift)) = logistic(-k*shift)
    # 比如 shift=4, k=8 => logistic(-32) ~ 1.266e-14, 非常接近0
    f_left = logistic(k * (x - shift))   # ~ (0, 1)

    # ------- 右段：x>1 -------
    # 当 x=1 时 logistic(k*((1-1)+ shift))= logistic(k*shift)= logistic(8*4)= logistic(32)
    # logistic(32) ~ 1-1.266e-14 => 非常接近1
    # x越大，则越接近1
    f_right = logistic(k * ((x - 1) + shift))
    
    # ------- 中段：0 <= x <= 1 -------
    # 直接就是 f(x)= x

    # 下面用 torch.where 做分段选择
    # 第一步：区分 (x<0) 和 (x>=0)
    y = torch.where(x < 0, f_left, x)  # 在 x<0 用左段，否则先用 x
    # 第二步：再区分 (x>1)
    # 对 “否则” 那部分再去判断 x>1 的情况
    y = torch.where(x > 1, f_right, y)

    return y


# =============== 示例测试 ===============
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 在 [-2,3] 范围内采样
    xs = torch.linspace(-2, 3, steps=400)
    # 试一下 k=8, shift=4
    ys = piecewise_smooth_in01(xs, k=8.0, shift=4.0)

    xs_np = xs.numpy()
    ys_np = ys.numpy()

    plt.figure(figsize=(7,4))
    plt.plot(xs_np, ys_np, label='piecewise_smooth_in01')
    # 画一条 y=x 参考线
    plt.plot(xs_np, xs_np, 'r--', label='y=x')
    
    # 画出 x=0、x=1 的辅助线
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(1, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.xlim(-2,3)
    plt.legend()
    plt.title("Piecewise function: in [0,1] => y=x, outside => logistic saturations")
    plt.show()

    # 你可以修改 k, shift 再看曲线变化。
    # k 越大，两侧越快速逼近 0 或 1； shift 越大，x=0/1 时的函数值越接近 0/1。
