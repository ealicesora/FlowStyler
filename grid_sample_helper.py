import torch
import torch.nn.functional as F

def make_grids(target_img):
    B, C, H, W = target_img.size()
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1) + 0.5
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W) + 0.5
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()
    grid = grid.cuda()

    vgrid = grid.clone()
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W  ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H  ,1 ) -1.0
    vgrid = vgrid.permute(0 ,2 ,3 ,1).cuda()
    return vgrid


def make_grids_no_allign(target_img,scale=4,downSampleScale = 4):
    B, C, H, W = target_img.size()
    H = H * scale //downSampleScale
    W = W * scale //downSampleScale
    print(H)
    print(W)
    
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1) 
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W) 
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()
    grid = grid.cuda()

    vgrid = grid.clone()
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0
    vgrid = vgrid.permute(0 ,2 ,3 ,1).cuda()
    return vgrid

def make_grids_no_allign_centers(target_img,scale=4,downSampleScale = 4):
    B, C, H, W = target_img.size()
    H = H * scale //downSampleScale
    W = W * scale //downSampleScale
    print(H)
    print(W)
    
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1) 
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W) 
    xx = xx.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    yy = yy.view(1 ,1 ,H ,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float()
    grid = grid.cuda()

    vgrid = grid.clone()
    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W ,1 ) -1.0 # (-1.0,1.0)
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H ,1 ) -1.0
    vgrid = vgrid.permute(0 ,2 ,3 ,1).cuda()
    return vgrid



def grid_sample_custom(image, optical,**args):
    return torch.nn.functional.grid_sample(image, optical,**args)
    
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


# if __name__ == "__main__":
#     image = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).view(1, 3, 1, 3)
    

#     optical = torch.Tensor([0.9, 0.5, 0.6, -0.7]).view(1, 1, 2, 2)

#     print (grid_sample(image, optical))

#     print (F.grid_sample(image, optical, padding_mode='border', align_corners=True))