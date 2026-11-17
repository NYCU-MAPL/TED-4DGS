#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from utils.general_utils import quaternion_to_matrix, inverse_sigmoid, PILtoTorch
import open3d as o3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, d_a=0, d_f=0, visible_mask=None, is_training=False, step=0, mask=None, part_id=-1, coding_start=80000, test_idx=None, model_path=None, num_down_emb_c=None, feat_context=None):
    ## view frustum filtering for acceleration
    time_sub = 0
    time_ = viewpoint_camera.time
    part_ = time_ // 0.1

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    visible_mask = visible_mask

    if part_id == -1:
        anchor = pc.get_anchor[visible_mask]
    else:
        anchor = pc.get_all_anchor()[visible_mask]
    anchor_and_time = torch.cat([anchor, torch.ones_like(anchor[:, :1]) * time_], dim=1)
    anchor_and_part = torch.cat([anchor, torch.ones_like(anchor[:, :1]) * (time_ // 0.1 * 0.1)], dim=1)
    #
    if part_id == -1:
        feat1 = pc._anchor_feat[visible_mask]
        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]

    else:
        feat1 = pc.get_all_anchor_feat()[visible_mask]
        grid_offsets = pc.get_all_offset()[visible_mask]
        grid_scaling = pc.get_all_scaling()[visible_mask]
    feat = feat1[:, :50]
    binary_grid_masks = pc.get_mask(mask)[visible_mask]
    binary_temp_masks = pc.get_temp_mask(None if pc.hyper_mode else mask)[visible_mask][:, 0, :]  # hyper(DSGC): None

    mean_t, var_t = pc._temp_mask[:, 1:3, 0][visible_mask], pc._temp_mask[:, 3:5, 0][visible_mask]

    encoded_temp_feature = pc._temperal_anchor_feat[visible_mask][:, 0, :].squeeze()


    Q_feat = 1
    Q_temp_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2

    Q_temp_mean = 0.1
    Q_temp_var = 0.1
    
    feat_rate = 0
    bit_per_param = None
    bit_per_feats_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    bit_per_temp_feat_param = None
    bit_per_temp_var_param = None
    bit_per_temp_mean_param = None
    hyper_rate = None

    if is_training:
        if step > 10000 and step <= coding_start:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat * 1.1
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling * 1.1
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets * 1.1
            if step >40000 and step <= coding_start:
                encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat * 1.1
                if not pc.hyper_mode:
                    mean_t = mean_t + torch.empty_like(mean_t).uniform_(-0.5, 0.5) * Q_temp_mean
                    var_t = var_t + torch.empty_like(var_t).uniform_(-0.5, 0.5) * Q_temp_var
            if pc.hyper_mode and step > 60001 and step <= coding_start:
                mean_t = mean_t + torch.empty_like(mean_t).uniform_(-0.5, 0.5) * Q_temp_mean
                var_t = var_t + torch.empty_like(var_t).uniform_(-0.5, 0.5) * Q_temp_var

        if step > coding_start or step == -1:
            anchor = pc.get_anchor
            feat = pc._anchor_feat
            grid_offsets = pc._offset
            grid_scaling = pc.get_scaling
            encoded_temp_feature = pc._temperal_anchor_feat
            mean_t, var_t = pc._temp_mask[:, 1:3, 0], pc._temp_mask[:, 3:5, 0]

            mask_anchor = pc.get_mask_anchor(mask)
            mask_anchor_bool = mask_anchor.to(torch.bool)
            mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()
            binary_grid_masks = pc.get_mask(mask)

            binary_temp_masks = pc.get_temp_mask(mask)[:, 0, :].squeeze()
            temp_anchor_bool = binary_temp_masks.to(torch.bool)
            temp_anchor_bool = torch.logical_and(temp_anchor_bool, mask_anchor_bool)
            temp_anchor_rate = (temp_anchor_bool.sum() / temp_anchor_bool.numel()).detach()

            means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, pc.temp_feat_dim*pc.num_temp_feat, pc.temp_feat_dim*pc.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
            
            Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
            Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
            Q_temp_mean = Q_temp_mean * (1.1 + torch.tanh(Q_temp_mean_adj))
            Q_temp_var = Q_temp_var * (1.1 + torch.tanh(Q_temp_var_adj))

            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)
            encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat.unsqueeze(1)
            mean_t = mean_t + torch.empty_like(mean_t).uniform_(-0.5, 0.5) * Q_temp_mean
            var_t = var_t + torch.empty_like(var_t).uniform_(-0.5, 0.5) * Q_temp_mean

            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
            choose_idx = choose_idx & mask_anchor_bool
            feat_chosen = feat[choose_idx]
            grid_scaling_chosen = grid_scaling[choose_idx]
            grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
            mean_temp_chosen = mean_t[choose_idx]
            var_temp_chosen = var_t[choose_idx]
            
            means = means[choose_idx]
            scales = scales[choose_idx]
            mean_scaling = mean_scaling[choose_idx]
            scale_scaling = scale_scaling[choose_idx]
            mean_temp_mean = mean_temp_mean[choose_idx]
            scale_temp_mean = scale_temp_mean[choose_idx]
            mean_temp_var = mean_temp_var[choose_idx]
            scale_temp_var = scale_temp_var[choose_idx]

            mean_offsets = mean_offsets[choose_idx]
            scale_offsets = scale_offsets[choose_idx]

            Q_feat = Q_feat[choose_idx]
            Q_scaling = Q_scaling[choose_idx]
            Q_offsets = Q_offsets[choose_idx]
            
            
            list_of_means = list(torch.split(means, split_size_or_sections=pc.chcm_slices_list, dim=-1))
            list_of_scales = list(torch.split(scales, split_size_or_sections=pc.chcm_slices_list, dim=-1))

            feat_list = list(torch.split(feat_chosen, split_size_or_sections=pc.chcm_slices_list, dim=-1))

            for i in range(len(list_of_means)):
                if i == 0:
                    bit_feat = pc.entropy_gaussian.forward(feat_list[i], list_of_means[i], list_of_scales[i], Q_feat, pc._anchor_feat.mean())
                    bit_feats = bit_feat
                    decoded_feat = feat_list[i]
                else:
                    dmean, dscale = torch.split(pc.get_chcm_mlp_list[i-1](decoded_feat), split_size_or_sections=[pc.chcm_slices_list[i], pc.chcm_slices_list[i]], dim=-1)
                    mean = list_of_means[i] + dmean
                    scale = list_of_scales[i] + dscale
                    bit_feat = pc.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat, pc._anchor_feat.mean())
                    bit_feats = torch.cat([bit_feats, bit_feat], dim=-1)
                    decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)

            if pc.chcm_for_scaling:
                dmean_scaling, dscale_scaling = torch.split(pc.get_chcm_mlp_scaling(decoded_feat), split_size_or_sections=[6, 6], dim=-1)
                mean_scaling = mean_scaling + dmean_scaling
                scale_scaling = scale_scaling + dscale_scaling
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            

            if pc.chcm_for_offsets:
                dmean_offsets, dscale_offsets = torch.split(pc.get_chcm_mlp_offsets(decoded_feat), split_size_or_sections=[3*pc.n_offsets, 3*pc.n_offsets], dim=-1)
                mean_offsets = mean_offsets + dmean_offsets
                scale_offsets = scale_offsets + dscale_offsets
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
            binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3*pc.n_offsets)
            bit_offsets = bit_offsets * binary_grid_masks_chosen
            
            Q_temp_mean = Q_temp_mean[choose_idx]
            Q_temp_var = Q_temp_var[choose_idx]
            bit_temp_mean = pc.entropy_gaussian.forward(mean_temp_chosen, mean_temp_mean, scale_temp_mean, Q_temp_mean, mean_t.mean())
            bit_temp_var = pc.entropy_gaussian.forward(var_temp_chosen, mean_temp_var, scale_temp_var, Q_temp_var, var_t.mean())

            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
            choose_idx = choose_idx & mask_anchor_bool & temp_anchor_bool
            temp_feat_chosen = encoded_temp_feature[choose_idx].view(-1, pc.temp_feat_dim*pc.num_temp_feat)

            mean_temp_feat = mean_temp_feat[choose_idx]
            scale_temp_feat = scale_temp_feat[choose_idx]
            Q_temp_feat = Q_temp_feat[choose_idx]

            binary_temp_masks_chosen = binary_temp_masks[choose_idx].unsqueeze(1)
            bit_temp_feat = pc.entropy_gaussian.forward(temp_feat_chosen, mean_temp_feat, scale_temp_feat, Q_temp_feat, pc._temperal_anchor_feat[temp_anchor_bool].mean())
            bit_temp_feat = bit_temp_feat * binary_temp_masks_chosen


            
            bit_per_temp_mean_param = torch.sum(bit_temp_mean) / bit_temp_mean.numel() * mask_anchor_rate
            bit_per_temp_var_param = torch.sum(bit_temp_var) / bit_temp_var.numel() * mask_anchor_rate
            bit_per_temp_feat_param = torch.sum(bit_temp_feat) / bit_temp_feat.numel() * mask_anchor_rate
            bit_per_feats_param = torch.sum(bit_feats) / bit_feats.numel() * mask_anchor_rate
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
            bit_per_param = (torch.sum(bit_feats) + torch.sum(bit_scaling) + torch.sum(bit_offsets) + torch.sum(bit_temp_mean) + torch.sum(bit_temp_var)) / \
                            (bit_feats.numel() + bit_scaling.numel() + bit_offsets.numel() + bit_temp_mean.numel() + bit_temp_var.numel()) * mask_anchor_rate + \
                                (torch.sum(bit_temp_feat) / bit_temp_feat.numel() * temp_anchor_rate)
            anchor = anchor[visible_mask]
            feat = feat[visible_mask]
            grid_offsets = grid_offsets[visible_mask]
            grid_scaling = grid_scaling[visible_mask]
            binary_grid_masks = binary_grid_masks[visible_mask]
            mask_anchor = mask_anchor[visible_mask]
            mask_anchor_bool = mask_anchor_bool[visible_mask]
            mean_t = mean_t[visible_mask]
            var_t = var_t[visible_mask]
            encoded_temp_feature = encoded_temp_feature[visible_mask].squeeze()
            binary_temp_masks = pc.get_temp_mask(mask)[visible_mask][:, 0, :]
    elif not pc.decoded_version:
        if step > coding_start or step == -1:
            feat_context = feat_context[visible_mask]
            means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, pc.temp_feat_dim*pc.num_temp_feat, pc.temp_feat_dim*pc.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
            
            Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
            Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
            Q_temp_mean = Q_temp_mean * (1.1 + torch.tanh(Q_temp_mean_adj))
            Q_temp_var = Q_temp_var * (1.1 + torch.tanh(Q_temp_var_adj))

            feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
            grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
            grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
            encoded_temp_feature = (STE_multistep.apply(encoded_temp_feature, Q_temp_feat, pc._temperal_anchor_feat.mean())).detach()
            mean_t = (STE_multistep.apply(mean_t, Q_temp_mean, mean_t.mean())).detach()
            var_t = (STE_multistep.apply(var_t, Q_temp_var, var_t.mean())).detach()
        else:
            if pc.hyper_mode:
                feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
                grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
                grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()

                if step > 40000:
                    encoded_temp_feature = (STE_multistep.apply(encoded_temp_feature, Q_temp_feat, pc._temperal_anchor_feat.mean())).detach()
                if step > 60001 and step <= coding_start:
                    mean_t = (STE_multistep.apply(mean_t, Q_temp_mean, mean_t.mean())).detach()
                    var_t = (STE_multistep.apply(var_t, Q_temp_var, var_t.mean())).detach()
            else:
                feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
                grid_scaling = (STE_multistep.apply(grid_scaling, 1.1 * Q_scaling, pc.get_scaling.mean())).detach()
                grid_offsets = (STE_multistep.apply(grid_offsets, 1.1 * Q_offsets, pc._offset.mean())).detach()

                if step > 40000:
                    encoded_temp_feature = (STE_multistep.apply(encoded_temp_feature, 1.1 * Q_temp_feat, pc._temperal_anchor_feat.mean())).detach()
                    mean_t = (STE_multistep.apply(mean_t, 1.1 * Q_temp_mean, mean_t.mean())).detach()
                    var_t = (STE_multistep.apply(var_t, 1.1 * Q_temp_var, var_t.mean())).detach()
    else:

        pass

    anchor += d_a[visible_mask]
    feat_w_time = d_f[visible_mask] + feat


    

    
    
    
    
    ob_view = anchor - viewpoint_camera.camera_center.cuda()
    cam_center = anchor - viewpoint_camera.camera_center.cuda() + anchor
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    if pc.hyper_mode:  # hyper(DSGC): color MLP input includes cam_center (+3 dims)
        cat_local_view = torch.cat([feat, ob_view, ob_dist, cam_center], dim=1)
        cat_local_view_w_time = torch.cat([feat_w_time, ob_view, ob_dist, cam_center], dim=1)
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]
        cat_local_view_w_time = torch.cat([feat_w_time, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]
    cat_local_global_view = torch.cat([feat, ob_view, ob_dist, cam_center], dim=1)  # [N_visible_anchor, 32+3+1+3]
    

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    if pc.hyper_mode:
        # hyper(DSGC): reshape first, existence after step>20000 with repeat
        neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
        if step > 20000:
            existence = pc.get_temporal_existence(time_, mean_t, var_t)
            existence = repeat(existence, 'n -> (n k)', k=pc.n_offsets)  # [N_visible_anchor*K]
            neural_opacity = neural_opacity * existence.view(-1, 1)  # [N_visible_anchor*K, 1]2
    else:
        if step > 0:
            existence = pc.get_temporal_existence(time_, mean_t, var_t)
            neural_opacity = neural_opacity * existence.unsqueeze(-1)  # [N_visible_anchor*K, 1]2
        neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_global_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view_w_time)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]
    

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]

    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    

    

    
    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets    

    
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])
    
    if pc.hyper_mode:
        scaling_for_stats = None
    else:
        full_scaling_for_stats = torch.zeros(anchor.shape[0] * pc.n_offsets, 3, device=scaling.device)
        full_scaling_for_stats[mask] = scaling
        scaling_for_stats = full_scaling_for_stats.view(anchor.shape[0], pc.n_offsets, 3)
    

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feats_param, bit_per_scaling_param, bit_per_offsets_param, bit_per_temp_feat_param, bit_per_temp_mean_param, bit_per_temp_var_param, feat_rate, hyper_rate, scaling_for_stats
    else:
        return xyz, color, opacity, scaling, rot, time_sub, feat_rate, hyper_rate

def render(viewpoint_camera, pc : GaussianModel, d_a, d_f, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0, mask=None, part_id=-1, coding_start=15000, test_idx=None, model_path=None, num_down_emb_c=None, feat_context=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feats_param, bit_per_scaling_param, bit_per_offsets_param, bit_per_temp_feat_param, bit_per_temp_mean_param, bit_per_temp_var_param, feat_rate, hyper_rate, scaling_for_stats = generate_neural_gaussians(viewpoint_camera, pc, d_a, d_f, visible_mask, is_training=is_training, step=step, mask=mask, part_id=part_id, coding_start=coding_start, test_idx=test_idx, model_path=model_path, num_down_emb_c=num_down_emb_c, feat_context=feat_context)
    else:
        xyz, color, opacity, scaling, rot, time_sub, feat_rate, hyper_rate = generate_neural_gaussians(viewpoint_camera, pc, d_a, d_f, visible_mask, is_training=is_training, step=step, mask=mask, part_id=part_id, coding_start=coding_start, test_idx=test_idx, model_path=model_path, num_down_emb_c=num_down_emb_c, feat_context=feat_context)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feats_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                "bit_per_temp_feat_param": bit_per_temp_feat_param,
                "bit_per_temp_var_param": bit_per_temp_var_param,
                "bit_per_temp_mean_param": bit_per_temp_mean_param,
                "feat_rate_per_param": feat_rate,
                "hyper_rate_per_param": hyper_rate,
                "scaling_for_stats": scaling_for_stats
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                "feat_rate_per_param": feat_rate,
                "hyper_rate_per_param": hyper_rate
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None, step=0, part_id=-1, d_a=0):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=1,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
    if part_id == -1:
        means3D = pc.get_anchor
    else:
        means3D = pc.get_all_anchor()

    means3D = means3D + d_a

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        if part_id == -1:
            scales = pc.get_scaling  # requires_grad = True
            rotations = pc.get_rotation  # requires_grad = True
        else:
            scales = pc.get_all_scaling()
            rotations = pc.get_all_rotation()

    
    
    radii_pure = rasterizer.visible_filter(
        means3D=means3D.to("cuda"),
        scales=scales[:, :3].to("cuda"),
        rotations=rotations.to("cuda"),
        cov3D_precomp=cov3D_precomp,  # None
    )

    
    if part_id == -1:
        return radii_pure > 0
    else:
        part_mask = pc.get_all_mask(part_id)
        return (radii_pure > 0) & (part_mask.squeeze(1))


def interpolate_tensor_with_max_time(a, t, max_time, temp_embeddings=0):
    """
    Interpolates the tensor `a` along the second dimension at time `t`.
    
    Parameters:
        a (torch.Tensor): Input tensor of shape [N, k, 10].
        t (float or torch.Tensor): Interpolation time, can be a scalar or a tensor of shape [N].
        max_time (float): The maximum time value corresponding to the last frame.
    
    Returns:
        torch.Tensor: Interpolated tensor of shape [N, 10].
    """
    N, k, _ = a.shape

    # Ensure t is a tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=a.device)

    # Map `t` to the range [0, k-1]
    t_normalized = (t / max_time) * (k - 1)

    # Handle the case when t == max_time explicitly
    if isinstance(t, torch.Tensor):
        is_max_time = (t == max_time).float()  # Mask for max_time
    else:
        is_max_time = float(t == max_time)  # Scalar case

    # Clamp t to the valid range [0, k-1]
    t_clamped = torch.clamp(t_normalized, 0, k - 1 - 1e-6)

    # Calculate indices t0 and t1
    t0 = t_clamped.floor().long()  # Lower index
    t1 = t0 + 1                    # Upper index

    # Calculate interpolation weights
    w1 = t_clamped - t0.float()    # Weight for t1
    w0 = 1 - w1                    # Weight for t0

    # Gather values from `a` at t0 and t1
    a_t0 = a[torch.arange(N), t0]  # Shape: [N, 10]
    a_t1 = a[torch.arange(N), t1]  # Shape: [N, 10]
    
    # Interpolation
    b_interpolated = w0.unsqueeze(-1) * a_t0 + w1.unsqueeze(-1) * a_t1  # Shape: [N, 10]

    # Use last frame for t == max_time
    b_max_time = a[:, -1, :]  # Shape: [N, 10]

    # Combine results
    b = b_interpolated * (1 - is_max_time).unsqueeze(-1) + b_max_time * is_max_time.unsqueeze(-1)

    return b

def temperal_embedding(value, L=3):
    """
    Apply Fourier features (positional embedding) to the input scalar.
    
    Args:
    - value: Input scalar (e.g., z coordinate) as a PyTorch tensor.
    - L: Number of frequencies to use (controls the dimension of embedding).
    
    Returns:
    - Embedded high-dimensional vector for the input scalar (PyTorch tensor).
    """
    embed = []
    embed.append(value)  # Add new dimension
    for i in range(L):
        embed.append(torch.sin(2**i * torch.pi * value))  # Add new dimension
        embed.append(torch.cos(2**i * torch.pi * value))  # Add new dimension
    return torch.cat(embed, dim=-1)  # Concatenate along the new dimension