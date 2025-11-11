import torch
import lib.Optical_Helper as OpticalHelper
def findBestMatch(target_index,datacollection,video_frames,masks_Optical_fwd,RaftModel,initvalue):
    # return initvalue
    with torch.no_grad():
        temp =initvalue
        temp_masks = torch.zeros_like(masks_Optical_fwd[0:1])
        for i in range(target_index-9,target_index-2):
            image1 = video_frames[i:1 + i]
            image2 = video_frames[target_index:target_index+1]
            special_flow = OpticalHelper.getRaftOptical(RaftModel,image1,image2)
            special_flow_bwd = OpticalHelper.getRaftOptical(RaftModel,image2,image1)
            masks = OpticalHelper.forward_backward_consistency_check(special_flow,special_flow_bwd,alpha= 0.1,beta=0.5)
            temp_masks = temp_masks + (1.0- masks[0])
            warped_data = OpticalHelper.warp(datacollection[i:1 + i],special_flow)*(1.0- masks_Optical_fwd[target_index-1]) * (1.0- masks[0])
            temp = temp + warped_data
        temp = temp/(temp_masks+1.0)
    return temp


# def findBestMatch_bidir(target_index,datacollection,video_frames,masks_Optical,RaftModel,initvalue,totalFramecounts, dir_fwd=True,windos = 15,endpadding =3):
#     # return initvalue
#     # careful

    
#     if dir_fwd:
#         LastAddingValue = -1
#         start_pos = target_index - windos # +1
#         start_pos = max(start_pos,0)
        
#         end_pos = target_index - endpadding
#         end_pos = max(end_pos,0)   
#     else:
#         LastAddingValue = 1
        
#          # +1
#         start_pos = target_index + endpadding
#         start_pos = min(start_pos,totalFramecounts)
        
#         end_pos = target_index + windos
#         end_pos = min(end_pos,totalFramecounts) 
    
#     if start_pos == end_pos:
#         return initvalue
#     # print(start_pos)
#     # print(end_pos)
#     with torch.no_grad():
#         temp = initvalue
#         temp_masks = torch.zeros_like(masks_Optical[0:1].cuda())
#         for i in range(start_pos,end_pos):
#             image1 = video_frames[i:1 + i].cuda()
#             image2 = video_frames[target_index:target_index+1].cuda()
#             special_flow = OpticalHelper.getRaftOptical(RaftModel,image1,image2)
#             special_flow_bwd = OpticalHelper.getRaftOptical(RaftModel,image2,image1)
#             masks = OpticalHelper.forward_backward_consistency_check(special_flow,special_flow_bwd,alpha= 0.01,beta=0.5)
#             temp_masks = temp_masks + (1.0- masks[0])
#             warped_data = OpticalHelper.warp(datacollection[i:1 + i].cuda(),special_flow,padding_mode='zeros')*(1.0- masks_Optical[target_index+LastAddingValue].cuda()) * (1.0- masks[0])
#             temp = temp + warped_data
#         temp = (temp)/(temp_masks+1.0)
#     return temp

def findBestMatch_bidir(
    target_index,
    datacollection,
    video_frames,
    masks_Optical,
    RaftModel,
    initvalue,
    totalFramecounts,
    dir_fwd=True,
    windos=15,
    endpadding=3,
    return_mask=False,
    mask_threshold=3
):
    # careful
    if dir_fwd:
        LastAddingValue = -1
        start_pos = max(target_index - windos, 0)
        end_pos = max(target_index - endpadding, 0)
    else:
        LastAddingValue = 1
        start_pos = min(target_index + endpadding, totalFramecounts)
        end_pos = min(target_index + windos, totalFramecounts)

    if start_pos == end_pos:
        if return_mask:
            return initvalue, None
        return initvalue

    candidate_indices = list(range(start_pos, end_pos))
    if len(candidate_indices) == 0:
        if return_mask:
            return initvalue, None
        return initvalue

    device = next(RaftModel.parameters()).device

    with torch.no_grad():
        initvalue_device = initvalue if initvalue.device == device else initvalue.to(device)

        source_frames = video_frames[candidate_indices].to(device, non_blocking=True).contiguous()
        target_frame = video_frames[target_index:target_index + 1].to(device, non_blocking=True).contiguous()
        target_frames = target_frame.expand(len(candidate_indices), -1, -1, -1).contiguous()

        fwd_flow = OpticalHelper.getRaftOptical(RaftModel, source_frames, target_frames)
        bwd_flow = OpticalHelper.getRaftOptical(RaftModel, target_frames, source_frames)

        fwd_occ, _ = OpticalHelper.forward_backward_consistency_check(
            fwd_flow,
            bwd_flow,
            alpha=0.01,
            beta=0.5
        )
        valid_masks = 1.0 - fwd_occ  # [B, H, W]

        data_batch = datacollection[candidate_indices].to(device, non_blocking=True).contiguous()
        warped_batch = OpticalHelper.warp(
            data_batch,
            fwd_flow,
            padding_mode='zeros'
        )

        occlusion_mask = 1.0 - masks_Optical[target_index + LastAddingValue].to(device, non_blocking=True)
        if occlusion_mask.dim() == 3:
            occlusion_mask = occlusion_mask.unsqueeze(1)  # [1, 1, H, W]

        valid_masks_expanded = valid_masks.unsqueeze(1)  # [B, 1, H, W]

        weighted_batch = warped_batch * valid_masks_expanded * occlusion_mask
        aggregated = weighted_batch.sum(dim=0, keepdim=True)  # [1, C, H, W]

        temp_masks = valid_masks.sum(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
        temp = (initvalue_device + aggregated) / (temp_masks + 0.01)

    if return_mask:
        mask_result = (temp_masks * occlusion_mask) > (mask_threshold - 0.1)
        return temp, mask_result

    return temp

from matplotlib import pyplot as plt

def visualize_c_long(c_long_map, save_path):
    """可视化c_long权重图（添加颜色条）"""
    plt.figure(figsize=(10, 8))
    plt.imshow(c_long_map, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Long-term Weights (max={c_long_map.max():.2f})")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

import os

def findBestMatch_bidir_16(target_index, datacollection, video_frames, masks_Optical, RaftModel, initvalue, totalFramecounts, dir_fwd=True, windos=15, endpadding=3, return_mask=False, mask_threshold=3):
    # Edge case handling with explicit frame count check
    if dir_fwd:
        start_pos = max(target_index - windos, 0)
        end_pos = max(target_index - endpadding, 0)
        LastAddingValue = -1
    else:
        start_pos = min(target_index + endpadding, totalFramecounts - 1)  # -1 for 0-based index
        end_pos = min(target_index + windos, totalFramecounts - 1)
        LastAddingValue = 1

    # Fix range inclusion (Python's range is exclusive)
    if dir_fwd:
        frame_indices = range(end_pos, start_pos)  # Include end_pos
    else:
        frame_indices = range(start_pos, end_pos ) if start_pos <= end_pos else []

    # print(start_pos)
    # print(end_pos)

    if not frame_indices:
        return (initvalue, None) if return_mask else initvalue

    if start_pos == end_pos:
        if return_mask:
            return initvalue,None
        return initvalue

    with torch.no_grad():
        temp = torch.zeros_like(initvalue)
        sum_weights = torch.zeros_like(masks_Optical[0].cuda())
        sum_prev = torch.zeros_like(masks_Optical[0].cuda())
        
        # Process from nearest to farthest
        for i in reversed(list(frame_indices)):  # Explicit conversion to list for reversed
            # print('processing')
            # print(str(i)+''+str(target_index))
            # Optical flow calculation
            image1 = video_frames[i:i+1].cuda()
            image2 = video_frames[target_index:target_index+1].cuda()
            fwd_flow = OpticalHelper.getRaftOptical(RaftModel, image1, image2)
            bwd_flow = OpticalHelper.getRaftOptical(RaftModel, image2, image1)
            
            # Consistency check (ensure mask has correct dimensions)
            consistency_mask = OpticalHelper.forward_backward_consistency_check(
                fwd_flow, bwd_flow, alpha=0.01, beta=0.5
            )[0].squeeze(0)  # Remove batch dim if needed

            # Weight calculation (Eq.10 implementation)
            c_current = 1.0 - consistency_mask
            c_long = torch.clamp(c_current - sum_prev, min=0)
            

            
            # Occlusion handling (verify mask indexing)
            occlusion_mask = (1.0 - masks_Optical[target_index + LastAddingValue].cuda()).unsqueeze(0)  # Add batch dim
            warped_data = OpticalHelper.warp(datacollection[i:i+1].cuda(), fwd_flow, padding_mode='border')  # Prefer border padding
            
            # Accumulate with broadcasting
            weighted_data = warped_data * c_long * occlusion_mask
            # print(c_long.shape)
            temp += weighted_data.squeeze(0)  # Match temp dimension
            sum_weights += c_long
            sum_prev += c_current

        # Normalization with device alignment
        valid_mask = sum_weights > 0.01
        temp = torch.where(valid_mask, temp / (sum_weights + 1e-8), initvalue.cuda())
        
        if return_mask:
            long_term_mask = (sum_weights * occlusion_mask.squeeze(0)) > (mask_threshold - 0.1)
            return temp, long_term_mask
            
        return temp.cpu() if initvalue.is_cpu else temp  # Match initvalue device
    
    
# def precompute_long_term_contributions(target_index, datacollection, video_frames, masks_Optical, RaftModel, totalFramecounts, window=15):
#     # 初始化缓存字典
#     cache = {
#         'warped_features': [],
#         'c_long_weights': [],
#         'occlusion_masks': []
#     }
    
#     # 处理前向历史帧
#     with torch.no_grad():
#         sum_prev = torch.zeros_like(masks_Optical[0].cuda())
        
#         # 从最近到最远处理历史帧
#         for j in reversed(range(max(0, target_index-window), target_index)):
#             # 计算双向光流
#             image1 = video_frames[j:j+1].cuda()
#             image2 = video_frames[target_index:target_index+1].cuda()
#             fwd_flow = OpticalHelper.getRaftOptical(RaftModel, image1, image2)
#             bwd_flow = OpticalHelper.getRaftOptical(RaftModel, image2, image1)
            
#             # 一致性检查
#             consistency_mask = OpticalHelper.forward_backward_consistency_check(
#                 fwd_flow, bwd_flow, alpha=0.01, beta=0.5
#             )[0].squeeze(0)
            
#             # 计算长期权重（公式10）
#             c_current = 1.0 - consistency_mask
#             c_long = torch.clamp(c_current - sum_prev, min=0)
#             sum_prev += c_current  # 累积历史权重
            
#             # 计算遮挡掩码
#             occlusion_mask = (1.0 - masks_Optical[target_index].cuda()).unsqueeze(0)
            
#             # 保存预处理结果
#             warped_feature = OpticalHelper.warp(datacollection[j:j+1].cuda(), fwd_flow)
#             cache['warped_features'].append(warped_feature)
#             cache['c_long_weights'].append(c_long * occlusion_mask)
    
#     return cache


# def precompute_long_term_contributions(target_index, datacollection, video_frames, masks_Optical, RaftModel, totalFramecounts, window=15):
#     cache = {
#         'warped_features': [],
#         'c_long_weights': [],
#         'occlusion_masks': [],
#         'debug_data': []  # 新增调试存储
#     }
    
#     with torch.no_grad():
#         sum_prev = torch.zeros_like(masks_Optical[0].cuda())
        
#         for j in reversed(range(max(0, target_index-window), target_index)):
#             # ========== 光流计算 ==========
#             image1 = video_frames[j:j+1].cuda()
#             image2 = video_frames[target_index:target_index+1].cuda()
            
#             # 保存原始帧对比
#             cache['debug_data'].append({
#                 'frame_j': image1.cpu().numpy(),
#                 'frame_target': image2.cpu().numpy()
#             })
            
#             # 计算双向光流（添加光流可视化）
#             fwd_flow = OpticalHelper.getRaftOptical(RaftModel, image1, image2)
#             bwd_flow = OpticalHelper.getRaftOptical(RaftModel, image2, image1)
            
#             # ========== 一致性检查 ==========
#             consistency_mask = OpticalHelper.forward_backward_consistency_check(
#                 fwd_flow, bwd_flow, alpha=0.01, beta=0.5
#             )[0].squeeze(0)
            
#             # 保存一致性掩码
#             cache['debug_data'][-1]['consistency_mask'] = consistency_mask.cpu().numpy()
            
#             # ========== 权重计算 ==========
#             c_current = 1.0 - consistency_mask
#             c_long = torch.clamp(c_current - sum_prev, min=0)
            
#             # 保存权重中间结果
#             cache['debug_data'][-1].update({
#                 'c_current': c_current.cpu().numpy(),
#                 'sum_prev_before': sum_prev.clone().cpu().numpy(),
#                 'c_long': c_long.cpu().numpy()
#             })
            
#             sum_prev += c_current  # 更新累积权重
            
#             # ========== 遮挡处理 ==========
#             occlusion_mask = (1.0 - masks_Optical[target_index].cuda()).unsqueeze(0)
            
#             # 保存遮挡掩码
#             cache['debug_data'][-1]['occlusion_mask'] = occlusion_mask.squeeze().cpu().numpy()
            
#             # ========== 特征变形 ==========
#             warped_feature = OpticalHelper.warp(datacollection[j:j+1].cuda(), fwd_flow)
            
#             # 保存变形前后的对比
#             cache['debug_data'][-1].update({
#                 'original_feature': datacollection[j:j+1].cpu().numpy(),
#                 'warped_feature': warped_feature.cpu().numpy()
#             })
            
#             # 保存最终结果
#             cache['warped_features'].append(warped_feature)
#             cache['c_long_weights'].append((c_long * occlusion_mask))
    
#     return cache

# def compute_long_term_loss(current_frame, cache):
#     """
#     论文公式7：L_temporal = 1/D * Σ(c_long * (x - warped_prev)^2)
#     """
#     loss = 0.0
#     D = current_frame.numel()  # 总像素数
    
#     for warped, weight in zip(cache['warped_features'], cache['c_long_weights']):
#         # 计算加权平方差
#         diff = (current_frame.cuda() - warped).pow(2)
#         weighted_diff = weight * diff
#         loss += weighted_diff.sum()
    
#     return loss #/ D



import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# 假设其他辅助函数和类已定义

def precompute_long_term_contributions(target_index, datacollection, video_frames, masks_Optical, RaftModel, totalFramecounts, window=15):
    """
    修正版函数:
    - 修正了 `final_weight` 扩展时的维度错误。
    """
    cache = {
        'warped_features': [],
        'c_long_weights': []
    }
    
    _, C, H, W = datacollection.shape
    composite_feature_image = torch.zeros(1, C, H, W).cuda()
    # window = 1
    with torch.no_grad():
        # 确保 sum_prev 的 shape 是 [B, H, W] 或 [B, 1, H, W]
        # 假设 masks_Optical[0] 是 [1, H, W]
        sum_prev = torch.zeros_like(masks_Optical[0].unsqueeze(0).cuda())

        for j in reversed(range(max(0, target_index - window), max(0, target_index - 0))):
            image1 = video_frames[j:j+1].cuda()
            image2 = video_frames[target_index:target_index+1].cuda()
            
            fwd_flow = OpticalHelper.getRaftOptical(RaftModel, image1, image2)
            bwd_flow = OpticalHelper.getRaftOptical(RaftModel, image2, image1)
            
            # 假设 consistency_mask 返回 [B, H, W], e.g., [1, 432, 768]
            consistency_mask = OpticalHelper.forward_backward_consistency_check_with_boundary(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)[0]
            
            c_current = 1.0 - consistency_mask
            c_long = torch.clamp(c_current - sum_prev, min=0)
            sum_prev += c_current
            
            # 假设 masks_Optical[target_index] 返回 [B, H, W], e.g., [1, 432, 768]
            occlusion_mask = 1.0 - masks_Optical[target_index].cuda() 
            # occlusion_mask =  masks_Optical[target_index].cuda() * 1.0
            warped_feature = OpticalHelper.warp(datacollection[j:j+1].cuda(), fwd_flow)
            
            # final_weight 的 shape 会是 [B, H, W], e.g., [1, 432, 768]
            final_weight = c_long # * occlusion_mask
            
            cache['warped_features'].append(warped_feature)
            # 保存的 weight 是 [B, H, W]
            cache['c_long_weights'].append(final_weight)
            
            # ========== 修正部分 ==========
            # warped_feature: [1, C, H, W]
            # final_weight:   [1, H, W]
            # 我们需要将 final_weight 变成 [1, C, H, W] 来作为蒙版
            # 方法: unsqueeze(1) -> [1, 1, H, W], 然后 expand_as
            mask_for_composition = final_weight.squeeze(0).expand_as(warped_feature)
            
            composite_feature_image = mask_for_composition * warped_feature + \
                                      (1 - mask_for_composition) * composite_feature_image

    save_path = f"debug_composite_feature_{target_index}.png"
    # print("save_path")
    save_image(composite_feature_image.squeeze(0), save_path, normalize=False)
    print(f"Debug composite feature image saved to {save_path}")
    
    return cache

def compute_long_term_loss(current_frame, cache):
    """
    修正了 loss 计算中 weight 张量的维度错误。
    """
    loss = 0.0
    current_frame_cuda = current_frame.cuda()

    for warped, weight in zip(cache['warped_features'], cache['c_long_weights']):
        # warped(diff) shape: [1, C, H, W]
        # weight shape: [1, H, W]
        diff = (current_frame_cuda - warped).pow(2)
        
        # ========== 修正部分 ==========
        # 使用 unsqueeze(1) 将 weight 从 [1, H, W] 变成 [1, 1, H, W]
        # 这样它就可以通过广播机制与 [1, C, H, W] 的 diff 相乘
        weighted_diff = weight.unsqueeze(1) * diff
        loss += weighted_diff.mean()
    
    return loss
