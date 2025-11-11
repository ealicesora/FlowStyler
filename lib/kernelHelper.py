import torch


def cubicspline_part(q):

    q = torch.abs(q)

    # q_clamp = torch.clamp(q, 0, 2)
    
    q_clamp = q
    
    part1 = 0.5 * q_clamp ** 3 - q_clamp ** 2 + 2.0/3.0
    part2 = 1.0 / 6.0 * (2 - q_clamp) ** 3
    
    result = torch.where(q_clamp <= 1.0, part1, part2)

    result = torch.where(q > 2, torch.zeros_like(q), result)
    
    return result

def cubic_spline(dx,dy,kernelsize = 2.0):

    modified_ksize= kernelsize / 2.0
    dx = torch.abs(dx/modified_ksize)
    dy = torch.abs(dy/modified_ksize)
    # print(dx)
    # q_clamp = torch.clamp(q, 0, 2)

    x_weight = cubicspline_part(dx)
    y_weight = cubicspline_part(dy)
    
    return x_weight * y_weight

def spheric_cubic_spline(dx,dy,kernelsize = 2.0):
    modified_ksize= kernelsize / 2.0
    dx = torch.abs(dx/modified_ksize)
    dy = torch.abs(dy/modified_ksize)
    h = torch.sqrt(dx*dx + dy*dy+ 0.00001)
    return cubicspline_part(h)

def linear_spline(dx,dy,kernelsize):
    x_weight = (1.0-dx)/kernelsize
    y_weight = (1.0-dy)/kernelsize
    x_weight = torch.clamp(x_weight,0.0,1.0)
    y_weight = torch.clamp(y_weight,0.0,1.0)
    return y_weight * x_weight

def KernelMethod(dx,dy,kernelsize,kerneltype):
    if kerneltype == 'linear':
        return linear_spline(dx,dy,kernelsize)
    if kerneltype == 'cubic':
        return cubic_spline(dx,dy,kernelsize)
    if kerneltype == 'spheric_cubic':
        return spheric_cubic_spline(dx,dy,kernelsize)