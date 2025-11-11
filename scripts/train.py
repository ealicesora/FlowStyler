#!/usr/bin/env python3
"""
FlowStyler Video Style Transfer - Main Training Script
"""
import sys
import yaml
from pathlib import Path
import torch
from torchvision import transforms

# Add module search paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.video_loader import VideoLoader
from src.data.optical_flow import OpticalFlowComputer
from src.models.lagrangian_field import LagrangianFieldSystem
from src.models.stylizer import Stylizer
import imager_IO_utils as utils
import lib.ColorMatchHelper as CMHelper
import lib.PreviousFrameMatchingHelper as MatchingHelper
from lib.ds_vectorFieldSmooth import l2_regularization, tv_regularization, huber_regularization, jacobian_regularization_advanced

try:
    from lib.SmoothOptimizer import Adam as SmoothAdam
except ImportError:  # pragma: no cover - SmoothOptimizer optional
    SmoothAdam = None

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path='config/default.yaml'):
    """Main entry point"""
    print("=" * 50)
    print("FlowStyler Video Style Transfer System")
    print("=" * 50)
    
    # Set default device to GPU when available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice in use: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 1. Load configuration
    print("\n[1/8] Loading configuration...")
    config = load_config(config_path)
    print(f"  Video: {config['video']['name']}")
    print(f"  Style: {config['style']['image_name']}")
    
    # 2. Load video
    print("\n[2/8] Loading video frames...")
    video_dir = Path(config['video']['data_dir']) / config['video']['name']
    
    # Handle resolution resampling options
    target_size = None
    keep_aspect_ratio = True
    if config['video'].get('enable_resize', False):
        target_size = (
            config['video']['target_height'],
            config['video']['target_width']
        )
        keep_aspect_ratio = config['video'].get('keep_aspect_ratio', True)
        print(f"  Enable resolution resampling: {target_size[0]}x{target_size[1]}")
        print(f"  Keep aspect ratio: {keep_aspect_ratio}")
    
    loader = VideoLoader(video_dir, target_size=target_size, keep_aspect_ratio=keep_aspect_ratio)
    frames = loader.load_all_frames()
    frames_uncolored = frames.clone()
    
    # Display dimension info
    size_info = loader.get_size_info()
    if size_info['resized']:
        print(f"  Original size: {size_info['original_size'][0]}x{size_info['original_size'][1]}")
        print(f"  Resampled size: {size_info['target_size'][0]}x{size_info['target_size'][1]}")
    print(f"  Loaded {len(frames)} frames: {frames.shape}")
    
    # 3. Load style image
    print("\n[3/8] Loading style image...")
    style_img = utils.image_loader(
        config['style']['image_dir'] + config['style']['image_name'],
        resize_arf=config['style']['resize_arf'],
        half_resize=True
    )
    print(f"  Style image tensor: {style_img.shape}")
    
    # 4. Compute optical flow
    print("\n[4/8] Computing optical flow...")
    flow_computer = OpticalFlowComputer(config)
    optical_fwd, optical_bwd, masks_fwd, masks_bwd = flow_computer.compute_all_flows(frames)
    
    # Process masks
    masks_fwd, masks_bwd = flow_computer.process_masks(
        masks_fwd, masks_bwd,
        iterations=config['optical_flow']['erode_iterations']
    )
    print(f"  Optical flow: {optical_fwd.shape}")
    print(f"  Occlusion masks: {masks_fwd.shape}")
    
    # Keep tensors on GPU for efficiency
    optical_fwd = optical_fwd.to(device)
    optical_bwd = optical_bwd.to(device)
    masks_fwd = masks_fwd.to(device)
    masks_bwd = masks_bwd.to(device)
    

    # 5. Color matching (optional)
    if config['color_matching']['enable']:
        print("\n[5/8] Performing color matching...")
        with torch.no_grad():
            # Use the first frame to compute color transform matrix
            first_frame = frames_uncolored[0].unsqueeze(0)  # [1, 3, H, W]
            alpha = config['color_matching']['alpha']
            beta = config['color_matching']['beta']
            _, color_tr = CMHelper.match_colors_for_image_set_ds(first_frame, style_img, alpha=alpha, beta=beta)
            # Apply to every frame
            for i in range(len(frames)):
                frames[i] = CMHelper.apply_CT(frames[i].unsqueeze(0), color_tr).squeeze(0)
            frames.clamp_(0.01, 1.0)
            print(f"  Color matching complete (alpha={alpha}, beta={beta})")
            del color_tr
    else:
        print("\n[5/8] Skipping color matching")

    # 6. Initialize models
    print("\n[6/8] Initializing models...")
    content_image = frames[0].unsqueeze(0).to(device)
    
    field_system = LagrangianFieldSystem(content_image, config)
    stylizer = Stylizer(style_img, content_image, config)
    
    print("  Field system initialized")
    print("  Stylizer initialized")
    
    # 7. Optimization loop
    print("\n[7/8] Starting optimization...")
    total_frames = len(frames)
    opt_cfg = config['optimization']
    strategy_cfg = config['strategy']
    reg_cfg = config['regularization']
    
    warp_field = field_system.warp_field
    alpha_field = field_system.alpha_field
    lagrangian_field = field_system.larg_field
    alphamode = config['fields']['alpha_mode']
    all_fields = [warp_field, alpha_field]
    
    disable_warping = opt_cfg.get('disable_warping', False)
    
    # Initialize learning-rate scaling (overridable for ablations)
    warp_field.lr_scale = config['fields']['warp_lr_scale']
    alpha_field.lr_scale = config['fields']['alpha_lr_scale']

    if disable_warping:
        warp_field.lr_scale = 0.0
    
    warp_collections = warp_field.getData().detach().clone().repeat(total_frames, 1, 1, 1).to(device)
    alpha_collections = alpha_field.getData().detach().clone().repeat(total_frames, 1, 1, 1).to(device)
    
    used_lr_base = opt_cfg['lr'] * opt_cfg.get('lr_multiplier', 1.0)
    backward_lr_scale = opt_cfg.get('backward_lr_scale', 0.01)
    total_neighbors = opt_cfg.get('total_neighbors', 1)
    num_steps_default = opt_cfg.get('num_steps', 40)
    num_steps_first_epoch = opt_cfg.get('num_steps_first_epoch', num_steps_default)
    num_steps_first_frame = opt_cfg.get('num_steps_first_frame', num_steps_default)
    num_steps_post_epoch = opt_cfg.get('num_steps_post_epoch', num_steps_default)
    post_epoch_threshold = opt_cfg.get('post_epoch_threshold', None)
    first_frame_restart_ratio = opt_cfg.get('first_frame_restart_ratio', 70)
    epochs_start_refine = opt_cfg.get('epochs_start_refine', -2)
    only_propagate = opt_cfg.get('only_propagate', False)
    backward_low_lr_mode = opt_cfg.get('backward_low_lr_mode', False)
    enable_padding_square = opt_cfg.get('enable_padding_to_square', False)
    loss_scales = opt_cfg.get('loss_scales', [1.0, 1.0, 1.0])
    log_interval = opt_cfg.get('log_interval', 10)
    enable_global_random_perspective = opt_cfg.get('enable_global_random_perspective', False)
    seperate_flip = opt_cfg.get('seperate_flip', True)
    save_intermediate = opt_cfg.get('save_intermediate', False)
    output_mask = opt_cfg.get('output_mask', False)
    step_atten_alpha_base = opt_cfg.get('step_atten_alpha', 1.02)
    optimizer_type = opt_cfg.get('optimizer', 'adamw').lower()
    matrix_reg_multiplier = opt_cfg.get('matrix_reg_multiplier', 1.0)
    
    enable_masks = strategy_cfg.get('enable_masks', True)
    enable_previous_matching = strategy_cfg.get('enable_previous_matching', False) and enable_masks
    previous_matching_cfg = strategy_cfg.get('previous_matching', {})
    match_window = previous_matching_cfg.get('window', 10)
    match_endpadding = previous_matching_cfg.get('endpadding', 2)
    match_mask_threshold = previous_matching_cfg.get('mask_threshold', 3)
    enable_our_opt = strategy_cfg.get('enable_our_opt', True)
    enable_occlusion_aware = strategy_cfg.get('enable_occlusion_aware', True)
    occlusion_weights_cfg = strategy_cfg.get('occlusion_base_weight', {})
    occlusion_alpha_weight = occlusion_weights_cfg.get('alpha', 1e-1)
    occlusion_warp_weight = occlusion_weights_cfg.get('warp', 1e-2)
    smooth_sigma = strategy_cfg.get('smooth_sigma', 1.0)
    smooth_radius = strategy_cfg.get('smooth_radius', 5)
    
    flow_reg_scale = reg_cfg.get('flow_reg_scale', 0.1)
    
    stylizer.set_weights(
        style_weight=config['style']['weight'],
        content_weight=config['style']['content_weight']
    )
    stylizer.set_random_perspective(enable_global_random_perspective)
    
    blur_for_grad = transforms.GaussianBlur(smooth_radius, smooth_sigma)
    blur_for_mask = transforms.GaussianBlur(5, 1.0)
    
    def pad_content_image(image: torch.Tensor) -> torch.Tensor:
        if not enable_padding_square:
            return image
        height_diff = image.shape[-1] - image.shape[-2]
        if height_diff <= 0:
            return image
        highest_half = height_diff // 2
        if highest_half == 0:
            return image
        padding = torch.ones((1, 3, highest_half, image.shape[-1]), device=image.device)
        return torch.cat([padding, image, padding], dim=2)
    
    def build_param_groups(used_lr_value: float):
        groups = []
        for field in all_fields:
            lr_scale = used_lr_value * field.lr_scale
            groups.append({'params': field.get_parameter()[0], 'lr': lr_scale})
        return groups
    
    def init_optimizer(used_lr_value: float):
        param_groups = build_param_groups(used_lr_value)
        if optimizer_type == 'smoothadam':
            if SmoothAdam is None:
                raise RuntimeError("Configuration requests SmoothAdam, but it is not available in the current environment.")
            return SmoothAdam(param_groups)
        return torch.optim.AdamW(param_groups)
    
    def smooth_field_grad(field):
        grad = field.output_ori.grad
        if grad is None:
            return
        smoothed = blur_for_grad(grad)
        update_mask = grad.abs() > 0.0
        smoothed[~update_mask] = grad[~update_mask]
        field.output_ori.grad = smoothed
    
    def build_mask_for_field(field, mask_tensor):
        if mask_tensor is None or not enable_masks:
            return torch.ones_like(field.output_ori[0, 0])
        mask_2d = mask_tensor.squeeze(0)
        translated = field.translateShapeMatchThisField(mask_2d.unsqueeze(0)).squeeze(0)
        return translated.clamp(0.0, 1.0)
    
    def apply_mask_to_grad(field, mask_tensor, base_weight, step_weight, previous_mask):
        if field.output_ori.grad is None or mask_tensor is None:
            return
        mask_2d = mask_tensor.squeeze()
        if mask_2d.dim() == 0:
            return
        working = 1.0 - mask_2d
        if previous_mask is not None:
            prev = previous_mask.squeeze()
            if prev.dim() > 2:
                prev = prev[0]
            working = working - prev * 1.0
            working = (working > 0.5).float()
        reweighted = (working + base_weight * step_weight).clamp(0.0, 1.0)
        while reweighted.dim() > 2 and reweighted.size(0) == 1:
            reweighted = reweighted.squeeze(0)
        if reweighted.dim() > 2:
            reweighted = reweighted.mean(0)
        if reweighted.dim() == 1:
            reweighted = reweighted.unsqueeze(0)
        blur_input = reweighted.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        reweighted = blur_for_mask(blur_input).squeeze(0).squeeze(0)
        translated = field.translateShapeMatchThisField(reweighted.unsqueeze(0)).squeeze()
        translated = translated.unsqueeze(0).unsqueeze(0)
        field.output_ori.grad = field.output_ori.grad * translated
    
    def compute_style_loss_for_frame(frame_idx, print_loss=False):
        with torch.no_grad():
            content_img = frames_uncolored[frame_idx:frame_idx+1]
            padded = pad_content_image(content_img)
            lagrangian_field.setNewImage(frames[frame_idx:frame_idx+1])
        def compose_proxy():
            with torch.enable_grad():
                return field_system.compose_image()
        return stylizer.compute_loss(compose_proxy, padded.to(device), print_loss)
    
    raft_model = flow_computer.model
    
    def advect_field_and_blend(
        target_field,
        collections,
        current_index,
        optical_tensor,
        masks_tensor,
        dir_forward,
        blend_alpha,
        reshaped_mask
    ):
        nonlocal pre_matching_mask_weight
        neighbor_index = current_index + 1 if dir_forward else current_index - 1
        advected = target_field.advect_withOptical(
            optical_tensor[current_index],
            paddingmode='reflection',
            sample_mode='bicubic'
        )
        used_later = collections[neighbor_index:neighbor_index+1] * (1.0 - reshaped_mask)
        if enable_previous_matching:
            matched, match_mask = MatchingHelper.findBestMatch_bidir(
                neighbor_index,
                collections,
                frames_uncolored,
                masks_tensor,
                raft_model,
                used_later,
                total_frames,
                dir_fwd=dir_forward,
                endpadding=match_endpadding,
                windos=match_window,
                mask_threshold=match_mask_threshold,
                return_mask=True
            )
            used_later = matched
            pre_matching_mask_weight = match_mask
        new_field = (blend_alpha * advected + (1.0 - blend_alpha) * collections[neighbor_index:neighbor_index+1]) * reshaped_mask + used_later
        return new_field
    
    style_name = Path(config['style']['image_name']).stem
    video_name = config['video']['name']
    method_name = opt_cfg.get('intermediate_method_name', 'larg')
    intermediate_root = Path(config['video']['results_dir']) / video_name / method_name
    if save_intermediate and not (seperate_flip and opt_cfg.get('use_flip_mode', False)):
        intermediate_root.mkdir(parents=True, exist_ok=True)
    
    def save_intermediate_frame(epoch_idx, frame_idx, flip_flag):
        if not save_intermediate:
            return
        if seperate_flip and opt_cfg.get('use_flip_mode', False):
            save_root = intermediate_root / f"{epoch_idx:04d}"
        else:
            save_root = intermediate_root
        save_root.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            lagrangian_field.setNewImage(frames[frame_idx:frame_idx+1])
            output_img = field_system.compose_image()
        mask_tensor = None
        if not flip_flag:
            if masks_fwd.shape[0] > 0:
                mask_idx = min(max(frame_idx - 1, 0), masks_fwd.shape[0] - 1)
                mask_tensor = masks_fwd[mask_idx]
        else:
            if masks_bwd.shape[0] > frame_idx:
                mask_tensor = masks_bwd[frame_idx]
        if flip_flag:
            image_path = save_root / f"Fliped_frame_{frame_idx:04d}_{style_name}.png"
            mask_path = save_root / f"Fliped_Masked_frame_{frame_idx:04d}_{style_name}.png"
        else:
            image_path = save_root / f"frame_{frame_idx:04d}_{style_name}.png"
            mask_path = save_root / f"Masked_frame_{frame_idx:04d}_{style_name}.png"
        utils.save_Tensor_image(output_img, str(image_path))
        if output_mask and mask_tensor is not None:
            mask_vis = (1.0 - mask_tensor.unsqueeze(0)).repeat(1, 3, 1, 1)
            mask_vis[:, 2] = 0.0
            utils.save_Tensor_image((output_img + mask_vis * 0.3).clamp(0.0, 1.0), str(mask_path))
    
    epochs = opt_cfg['epochs']
    use_flip = opt_cfg.get('use_flip_mode', False)
    flip = use_flip
    pre_matching_mask_weight = None if enable_previous_matching else None
    
    for ep in range(epochs):
        if use_flip:
            flip = not flip
        direction = "backward" if flip else "forward"
        print(f"\n  Epoch {ep+1}/{epochs} ({direction})")
        
        stylizer.set_random_perspective(enable_global_random_perspective)
        
        if post_epoch_threshold is not None and ep > post_epoch_threshold:
            epoch_num_steps = num_steps_post_epoch
        else:
            epoch_num_steps = num_steps_default
        
        with torch.no_grad():
            if not flip:
                warp_field.updateDataForever(warp_collections[0:1])
                alpha_field.updateDataForever(alpha_collections[0:1])
            else:
                warp_field.updateDataForever(warp_collections[-1:])
                alpha_field.updateDataForever(alpha_collections[-1:])
        
        frame_indices = range(total_frames)
        if flip:
            frame_indices = range(total_frames - 1, -1, -1)
        
        for frame_idx in frame_indices:
            current_frame = frame_idx
            print(f"    Processing frame {current_frame+1}/{total_frames}...", end=' ')
            
            used_lr = used_lr_base
            if flip and backward_low_lr_mode:
                used_lr = used_lr_base * backward_lr_scale
            
            if ep == 0 and current_frame == 0:
                num_steps_current = num_steps_first_frame
                stylizer.set_random_perspective(False)
            else:
                if ep == 0:
                    num_steps_current = num_steps_first_epoch
                else:
                    num_steps_current = epoch_num_steps
                stylizer.set_random_perspective(enable_global_random_perspective)
            
            if only_propagate:
                if (not flip and current_frame != 0) or (flip and current_frame != total_frames - 1):
                    num_steps_current = 1
                    used_lr = 0.0
            
            optimizer_single = init_optimizer(used_lr)
            step = 0
            last_loss_value = 0.0
            
            while step <= num_steps_current:
                overall_loss_control = 0.0 if step == 0 else 1.0
                if ep == 0 and current_frame == 0 and first_frame_restart_ratio > 0:
                    if step > 0 and step % first_frame_restart_ratio == first_frame_restart_ratio - 1:
                        optimizer_single = init_optimizer(used_lr)
                        print("restarting optimizer", end=' ')
                
                if enable_our_opt and step == 1 and ep >= epochs_start_refine:
                    if ep != 0 or current_frame != 0:
                        warp_field.DumpPara_toOpt(optimizer_single)
                        alpha_field.DumpPara_toOpt(optimizer_single)
                
                loss = torch.tensor(0.0, device=device)
                frame_used_last = None
                for neighbor_idx in range(total_neighbors):
                    if flip:
                        frame_used = current_frame - neighbor_idx
                    else:
                        frame_used = current_frame + neighbor_idx
                    if frame_used < 0 or frame_used >= total_frames:
                        continue
                    frame_used_last = frame_used
                    weight_idx = min(neighbor_idx, len(loss_scales) - 1)
                    loss = loss + compute_style_loss_for_frame(
                        frame_used,
                        print_loss=(step % log_interval == 0)
                    ) * loss_scales[weight_idx]
                
                if total_neighbors != 1 and frame_used_last is not None:
                    if not flip and frame_used_last < optical_fwd.shape[0]:
                        warp_field.advect_withOptical_andUpdate(optical_fwd[frame_used_last])
                        alpha_field.advect_withOptical_andUpdate(optical_fwd[frame_used_last])
                    elif flip and frame_used_last < optical_bwd.shape[0]:
                        warp_field.advect_withOptical_andUpdate(optical_bwd[frame_used_last])
                        alpha_field.advect_withOptical_andUpdate(optical_bwd[frame_used_last])
                
                if flow_reg_scale != 0.0:
                    flow = warp_field.getData()
                    smooth_loss = torch.tensor(0.0, device=device)
                    if reg_cfg['lambda_l2'] != 0.0:
                        smooth_loss = smooth_loss + reg_cfg['lambda_l2'] * l2_regularization(flow)
                    if reg_cfg['lambda_tv'] != 0.0:
                        smooth_loss = smooth_loss + reg_cfg['lambda_tv'] * tv_regularization(flow)
                    if reg_cfg['lambda_huber'] != 0.0:
                        smooth_loss = smooth_loss + reg_cfg['lambda_huber'] * huber_regularization(flow, delta=1.0)
                    if reg_cfg.get('lambda_jacobian', 0.0) != 0.0:
                        smooth_loss = smooth_loss + reg_cfg['lambda_jacobian'] * jacobian_regularization_advanced(flow)
                    if smooth_loss != 0:
                        smooth_loss = smooth_loss * flow_reg_scale
                        if step % log_interval == 0:
                            print(f"smoothness loss: {smooth_loss.item():.6f}", end=' ')
                        loss = loss + smooth_loss
                
                matrix_coeff = reg_cfg['matrix_reg_weight'] * matrix_reg_multiplier
                if matrix_coeff != 0.0:
                    alpha_image = alpha_field.get_colored_reshaped(lagrangian_field.downsampledShape)
                    matrix_loss = lagrangian_field.getMatrixRegLoss(
                        alpha_image=alpha_image,
                        alphamode=alphamode,
                        print_loss=(step % log_interval == -1),
                        weight=matrix_coeff
                    )
                    loss = loss + matrix_loss
                
                loss = loss * overall_loss_control
                last_loss_value = float(loss.detach().item())
                
                if loss.grad_fn is not None:
                    loss.backward()
                
                smooth_field_grad(alpha_field)
                smooth_field_grad(warp_field)
                
                step_atten_alpha = step_atten_alpha_base ** step
                if enable_occlusion_aware and enable_our_opt and step > 0:
                    if not flip and current_frame != 0 and masks_fwd.shape[0] > 0:
                        mask_idx = min(current_frame - 1, masks_fwd.shape[0] - 1)
                        mask_tensor = masks_fwd[mask_idx]
                        apply_mask_to_grad(alpha_field, mask_tensor, occlusion_alpha_weight, step_atten_alpha, pre_matching_mask_weight)
                        apply_mask_to_grad(warp_field, mask_tensor, occlusion_warp_weight, step_atten_alpha, pre_matching_mask_weight)
                    elif flip and current_frame != total_frames - 1 and masks_bwd.shape[0] > current_frame + 1:
                        mask_idx = current_frame + 1
                        mask_tensor = masks_bwd[mask_idx]
                        apply_mask_to_grad(alpha_field, mask_tensor, occlusion_alpha_weight, step_atten_alpha, pre_matching_mask_weight)
                        apply_mask_to_grad(warp_field, mask_tensor, occlusion_warp_weight, step_atten_alpha, pre_matching_mask_weight)
                
                optimizer_single.step()
                optimizer_single.zero_grad()
                
                with torch.no_grad():
                    for field in all_fields:
                        field.reset()
                
                step += 1
            
            print(f"Loss: {last_loss_value:.6f}")
            
            with torch.no_grad():
                warp_collections[current_frame:current_frame+1].copy_(warp_field.output_ori)
                alpha_collections[current_frame:current_frame+1].copy_(alpha_field.output_ori)
                
                blend_alpha = 0.5 if ep < epochs_start_refine else 1.0
                if not flip and current_frame < masks_fwd.shape[0]:
                    mask_used = masks_fwd[current_frame]
                elif flip and current_frame < masks_bwd.shape[0]:
                    mask_used = masks_bwd[current_frame]
                else:
                    mask_used = None
                
                should_advect = ((not flip) and current_frame < total_frames - 1) or (flip and current_frame > 0)
                if should_advect:
                    if not flip:
                        opt_flow = optical_fwd[current_frame]
                        masks_tensor = masks_fwd
                        dir_forward = True
                    else:
                        opt_flow = optical_bwd[current_frame]
                        masks_tensor = masks_bwd
                        dir_forward = False
                    
                    mask_for_warp = build_mask_for_field(warp_field, mask_used)
                    mask_for_alpha = build_mask_for_field(alpha_field, mask_used)
                    
                    mask_for_state = mask_used if mask_used is not None else torch.ones(
                        (1, opt_flow.shape[-2], opt_flow.shape[-1]),
                        device=opt_flow.device
                    )
                    
                    warp_field.UpdateAdamPara_fromOpt(
                        optimizer_single,
                        opt_flow,
                        mask_for_state
                    )
                    new_warp = advect_field_and_blend(
                        warp_field,
                        warp_collections,
                        current_frame,
                        optical_bwd if flip else optical_fwd,
                        masks_tensor,
                        dir_forward,
                        blend_alpha,
                        mask_for_warp
                    )
                    warp_field.updateDataForever(new_warp)
                    
                    alpha_field.UpdateAdamPara_fromOpt(
                        optimizer_single,
                        opt_flow,
                        mask_for_state
                    )
                    new_alpha = advect_field_and_blend(
                        alpha_field,
                        alpha_collections,
                        current_frame,
                        optical_bwd if flip else optical_fwd,
                        masks_tensor,
                        dir_forward,
                        blend_alpha,
                        mask_for_alpha
                    )
                    alpha_field.updateDataForever(new_alpha)
                else:
                    warp_field.UpdateAdamPara_fromOpt(optimizer_single, no_advect=True)
                    alpha_field.UpdateAdamPara_fromOpt(optimizer_single, no_advect=True)
            
            save_intermediate_frame(ep, current_frame, flip)
    
    # 8. Render results
    print("\n[8/8] Rendering results...")
    output_frames = torch.zeros_like(frames).to(device)
    
    with torch.no_grad():
        for i in range(total_frames):
            field_system.warp_field.updateDataForever(
                warp_collections[i:i+1]
            )
            field_system.alpha_field.updateDataForever(
                alpha_collections[i:i+1]
            )
            field_system.update_particle_colors(frames[i].unsqueeze(0).to(device))
            
            output_frames[i] = field_system.compose_image().squeeze(0)
            
            if (i + 1) % 10 == 0:
                print(f"  Rendering progress: {i+1}/{total_frames}")
    
    # 9. Post-process and save
    print("\n[9/9] Post-processing and saving...")
    if config['postprocess']['smooth_brightness']:
        from src.utils.postprocess import temporal_brightness_smoothing
        output_frames = temporal_brightness_smoothing(
            output_frames,
            smooth_factor=config['postprocess']['smooth_factor']
        )
    
    # Save resized outputs
    save_dir = Path(config['video']['results_dir']) / \
               f"{config['video']['name']}_{config['style']['image_name']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    from src.utils.io import save_tensor_as_images, save_tensor_as_video
    save_tensor_as_images(output_frames, str(save_dir))
    
    # Save resized video result
    video_path = save_dir / "output_resized.mp4"
    save_tensor_as_video(output_frames, str(video_path), fps=config['video'].get('fps', 12))
    

    print(f"\n✅ Done! Results saved to: {save_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config)
