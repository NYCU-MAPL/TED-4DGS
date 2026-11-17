# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
from comet_ml import Experiment, ExistingExperiment
import subprocess
import copy
from torch.utils.data import DataLoader


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import anchor_round_digits, Q_anchor, encoder_anchor, get_binary_vxl_size, STE_multistep
from utils.encodings import STE_binary
from lpipsPyTorch import lpips
from utils.loader_utils import FineSampler, get_stamp_list
from torchvision.transforms import ToPILImage
from scene.regulation import compute_temp_feat_smoothness
from utils.extra_utils import o3d_knn, weighted_l2_loss_v2, image_sampler, calculate_distances, sample_camera
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from torchvision.utils import save_image
bit2MB_scale = 8 * 1024 * 1024
run_codec = True

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

import os
import re



def _service_network_gui(pipe, gaussians, background, source_path, iteration, opt):
    # network gui not available in scaffold-gs yet
    if network_gui.conn == None:
        network_gui.try_connect()
    while network_gui.conn != None:
        try:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, source_path)
            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                break
        except Exception as e:
            network_gui.conn = None


def _load_or_init_gaussians(gaussians, checkpoint, opt, load_iteration):
    first_iter = load_iteration if load_iteration is not None else 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        gaussians.training_setup(opt)
    return first_iter


def _maybe_setup_triplane(gaussians, iteration, opt):
    if iteration == opt.triplane_init_fit_iter:
        gaussians.set_feature_net()
        gaussians.training_setup_triplane(opt)


def _prep_and_render(gaussians, viewpoint_cam, iteration, opt, pipe, background, dataset, scene):
    """Shared per-camera forward for both hyper and n3dv: temp-feature noise, deformation, render.
    Returns (render_pkg, voxel_visible_mask, feat_rate)."""
    encoded_temp_feature = gaussians._temperal_anchor_feat[:, 0, :].squeeze()
    binary_temp_masks = gaussians.get_temp_mask()[:, 0, :]
    feat_context = None
    feat_rate = 0
    if iteration > 80000:
        Q_temp_feat = 1
        feat_context, feat_rate, f1rate, f2rate, f3rate = gaussians.feature_net(gaussians.get_anchor, itr=iteration)
        means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
        torch.split(feat_context, split_size_or_sections=[gaussians.feat_dim, gaussians.feat_dim, 6, 6, 3*gaussians.n_offsets, 3*gaussians.n_offsets, gaussians.temp_feat_dim*gaussians.num_temp_feat, gaussians.temp_feat_dim*gaussians.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
        Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
        encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat * 1.1
    elif iteration > 40000:
        Q_temp_feat = 1
        encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat * 1.1
    num_down_emb_c = dataset.min_embeddings
    d_a, d_f, d_o, d_s, _ = gaussians._deformation(time_emb=viewpoint_cam.time, embeddings=encoded_temp_feature, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_c)
    d_a = d_a * binary_temp_masks
    d_f = d_f * binary_temp_masks
    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, step=iteration, d_a=d_a)
    retain_grad = (iteration < opt.update_until and iteration >= 0)
    render_pkg = render(viewpoint_cam, gaussians, d_a, d_f, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration, mask=None, coding_start=opt.triplane_init_fit_iter, model_path=scene.model_path, num_down_emb_c=dataset.min_embeddings, feat_context=feat_context)
    return render_pkg, voxel_visible_mask, feat_rate


def training(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint, debug_from, wandb=None, logger=None, load_iteration=None):
    first_iter = 0
    hyper = (getattr(dataset, 'loader', '') == 'nerfies')
    # ICModule (per-view image enhancement): user opt-in via --use_ic_module. Not supported on hypernerf
    # (im logic is only wired into the n3dv render/optim/save path), so warn + disable if requested there.
    use_ic_module = bool(getattr(args_param, 'use_ic_module', False))
    if use_ic_module and hyper:
        print("[warning] --use_ic_module is not supported for hypernerf (nerfies) scenes; ignoring it.")
        use_ic_module = False
    dataset.use_ic_module = use_ic_module  # bridge effective value onto ModelParams namespace → recorded in cfg_args + read natively by GaussianModel(args=dataset)
    tb_writer = prepare_output_and_logger(dataset)
    attribute_cfg = dataset.attribute_config
    print("lambda:", args_param.lmbda)

    # ---- GaussianModel / Scene construction ----
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
        dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features, log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D, chcm_slices_list=args_param.chcm_slices_list,
        chcm_for_offsets=args_param.chcm_for_offsets, chcm_for_scaling=args_param.chcm_for_scaling,
        attribute_config=dataset.attribute_config, args=dataset)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, opt=opt)
    first_iter = _load_or_init_gaussians(gaussians, checkpoint, opt, load_iteration)

    gaussians.attribute_config["comp_iter"] = opt.triplane_init_fit_iter + 5000
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    prev_num_pts = 0
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=130)
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    mask = None
    mask_weight = None   # mask value used in render/report is always None for both paths (unified)

    # ---- camera / dataloader / other setup (#3: fully different between paths) ----
    if hyper:
        if not opt.dataloader:
            print("Not using dataloader")
            train_cams = scene.getTrainCameras()
        if opt.dataloader:
            train_cams = scene.getTrainCameras()
        num_traincams = 1
        cam_no_list = list(set(c.cam_no for c in train_cams))
        loss_list = np.zeros([num_traincams, scene.maxtime]) + 100  # pick frames not yet sampled
        viewpoint_stack = train_cams
    else:
        if not opt.dataloader:
            print("Not using dataloader")
            train_viewpoint_cams = scene.getTrainCameras().copy()
        if opt.dataloader:
            print("Using dataloader")
            viewpoint_stack = scene.getTrainCameras(data_loader=True)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=args_param.batch_size, shuffle=True, num_workers=16, collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        step = 300
        if len(viewpoint_stack) > step:
            sampled_views = [viewpoint_stack[i] for i in range(0, len(viewpoint_stack), step)]
        if checkpoint is None:
            checkpoint_path = None
        else:
            ckpt_path = args_param.ckpt_path
            checkpoint_path = os.path.join(ckpt_path, "point_cloud", "iteration_" + str(load_iteration), "mlp_checkpoints.pth")
        if use_ic_module:
            gaussians.residual_map_setup(viewpoint_stack[0].image_height, viewpoint_stack[0].image_width, len(sampled_views), opt, checkpoint_path=checkpoint_path)
        mask_update_frequency = 150
        mask_grad_accumulator = []

    background = None  # defensive: gui helper reads it before loop-body reassigns
    for iteration in range(first_iter, opt.iterations + 1):
        _service_network_gui(pipe, gaussians, background, dataset.source_path, iteration, opt)

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ---- Pick cameras (#2: different sampling) ----
        if hyper:
            frame_set = np.random.choice(range(math.ceil(len(viewpoint_stack) / 2)), size=max(opt.batch_size // 2, 1))
            viewpoint_cams = [viewpoint_stack[(f*2) % scene.maxtime] for f in frame_set] + \
                             [viewpoint_stack[(f*2+1) % scene.maxtime] for f in frame_set]
        else:
            if opt.dataloader == False:
                if not viewpoint_stack:
                    viewpoint_stack = list(range(len(train_viewpoint_cams)))
                viewpoint_cams = [train_viewpoint_cams[viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))]]
            else:
                try:
                    viewpoint_cams = next(loader)
                except StopIteration:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=args_param.batch_size, shuffle=True, num_workers=16, collate_fn=list)
                    loader = iter(viewpoint_stack_loader)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        _maybe_setup_triplane(gaussians, iteration, opt)

        # ---- init-time hooks (#5: different trigger timing) ----
        if hyper:
            if iteration == 20001:
                a = get_time_mean_var(iteration, scene, render, (pipe, background), weighted_mask=None, args_param=args_param, coding_start=opt.triplane_init_fit_iter, num_down_emb_c=dataset.min_embeddings)
        else:
            if iteration == 1:
                a = get_time_mean_var(iteration, scene, render, (pipe, background), weighted_mask=None, args_param=args_param, coding_start=opt.triplane_init_fit_iter, num_down_emb_c=dataset.min_embeddings)
            if iteration == 60001:
                gaussians.set_masking()

        # ---- render loop (#6: hyper accumulates a batch list / n3dv per-image + CRN) ----
        scaffold_loss_list = []
        feat_rate = 0
        if hyper:
            images = []
            gt_images = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            offset_selection_mask_list = []
            voxel_visible_mask_list = []
            opacity_list = []
            cam_no_list, frame_no_list = [], []
            time_list = []
            for viewpoint_cam in viewpoint_cams:
                if type(viewpoint_cam.original_image) == type(None):
                    viewpoint_cam.load_image()  # lazy loading (avoid OOM)
                cam_no = viewpoint_cam.cam_no
                frame_no = viewpoint_cam.frame_no
                cam_no_list.append(cam_no)
                frame_no_list.append(frame_no)
                render_pkg, voxel_visible_mask, feat_rate = _prep_and_render(gaussians, viewpoint_cam, iteration, opt, pipe, background, dataset, scene)
                image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                t = viewpoint_cam.time
                visibility_filter_list.append(visibility_filter)
                viewspace_point_tensor_list.append(viewspace_point_tensor)
                opacity_list.append(opacity)
                offset_selection_mask_list.append(offset_selection_mask)
                voxel_visible_mask_list.append(voxel_visible_mask)
                time_list.append(viewpoint_cam.time)
                ssim_loss = (1.0 - ssim(image, gt_image))
                scaling_reg = scaling.prod(dim=1).mean()
                scaffold_loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
                if opt.opacity_l1_coef_fine > 0:
                    scaffold_loss += opt.opacity_l1_coef_fine * torch.sigmoid(opacity.mean())
                scaffold_loss_list.append(scaffold_loss)
        else:
            for viewpoint_cam in viewpoint_cams:
                render_pkg, voxel_visible_mask, feat_rate = _prep_and_render(gaussians, viewpoint_cam, iteration, opt, pipe, background, dataset, scene)
                image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity, scaling_for_stats = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["scaling_for_stats"]
                original_image = viewpoint_cam.original_image.cuda()
                view_id = viewpoint_cam.colmap_id
                cor_view_id = view_id // 300
                if use_ic_module and iteration > opt.update_from:
                    enhanced_image, residual_map = gaussians.ic_modules[cor_view_id](image)
                    enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)
                gt_image = viewpoint_cam.original_image.cuda()
                if use_ic_module and iteration > opt.update_from:
                    Ll1 = l1_loss(enhanced_image, gt_image)
                    ssim_loss = (1.0 - ssim(enhanced_image, gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image)
                    ssim_loss = (1.0 - ssim(image, gt_image))
                scaling_reg = scaling.prod(dim=1).mean()
                scaffold_loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
                if opt.opacity_l1_coef_fine > 0:
                    scaffold_loss += opt.opacity_l1_coef_fine * torch.sigmoid(opacity.mean())
                scaffold_loss_list.append(scaffold_loss)

        scaffold_loss = torch.stack(scaffold_loss_list).mean()
        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]
        bit_per_temp_feat_param = render_pkg["bit_per_temp_feat_param"]
        

        if iteration % 2000 == 0 and bit_per_param is not None:
            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB
            with torch.no_grad():
                grid_masks = gaussians._mask.data
                binary_grid_masks = (torch.sigmoid(grid_masks) > 0.01).float()
                mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks + 0.0)

        # ---- loss (#4 + #7: different extra loss terms) ----
        loss = scaffold_loss
        if opt.triplane_init_fit_iter <= iteration and bit_per_param is not None:
            loss = loss + args_param.lmbda * bit_per_param
        if hyper:
            if args_param.temp_loss:
                temp_mask_loss = torch.sigmoid(gaussians._temp_mask[:, 0, :]).mean() * 0.000001
                loss += temp_mask_loss
            if iteration > 65000:
                mask_loss = torch.sigmoid(gaussians._mask).mean() * args_param.lmbda_mask
                loss += mask_loss
            weights = gaussians._deformation.weight
            N, C = weights.shape
            first_difference = weights[1:, :] - weights[:N-1, :]
            second_difference = first_difference[1:, :] - first_difference[:N-2, :]
            coef_tv_temporal_embedding = 0.0006
            loss += coef_tv_temporal_embedding * torch.square(second_difference).mean()
            # embedding reg using knn (Dynamic3DGaussians)
            if prev_num_pts != gaussians._anchor.shape[0]:
                neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._anchor.detach().cpu().numpy(), 20)
                neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
                neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
                neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
                prev_num_pts = gaussians._anchor.shape[0]
            mask_prob = gaussians.get_temp_mask(mask)[:, 0, :].float().squeeze(-1)
            m_i = mask_prob.unsqueeze(1).repeat(1, 20).unsqueeze(-1)
            m_j = mask_prob[neighbor_indices].unsqueeze(-1)
            reg_temp_mask_coef = 1
            loss += reg_temp_mask_coef * weighted_l2_loss_v2(m_i, m_j, neighbor_weight)
        else:
            if iteration > 60000:
                mask_loss = torch.sigmoid(gaussians._mask).mean() * args_param.lmbda_mask
                loss += mask_loss
            temp_mask_loss = torch.sigmoid(gaussians._temp_mask[:, 0, :]).mean() * 0.00002
            loss += temp_mask_loss
            if use_ic_module and iteration > opt.update_from:
                loss += 0.01 * torch.abs(residual_map).mean()

        loss.backward()
        iter_end.record()
        if not hyper:
            torch.nn.utils.clip_grad_norm_([gaussians._mask], max_norm=0.0003)

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            F_bits = 0
            if not hyper:
                if iteration > opt.triplane_init_fit_iter:
                    F_bits = feat_rate / bit2MB_scale
            if iteration % 10 == 0:
                if hyper:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "psnr": f"{psnr(gt_image, image).mean().double():.{2}f}", "points": f"{gaussians.get_mask(mask_weight).numel()}", "time": f"{gaussians.get_mask(mask_weight).sum()}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "psnr": f"{psnr(gt_image, image).mean().double():.{2}f}", "points": f"{gaussians.get_anchor.shape[0]}", "time": f"{gaussians.get_temp_mask().sum()}"})
                progress_bar.update(10)
            if not hyper:
                if iteration % 1000 == 0:
                    mask_anchor = scene.gaussians.get_mask_anchor()
                    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()
                    print(f"Mask rate: {mask_anchor_rate}")
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration % 100 == 0:
                a = "train"
                log = {
                        f'{a} loss': loss,
                        f'{a} L1_loss': Ll1,
                        f'{a} ssim': ssim_loss,
                        f'{a} scaling_reg': scaling_reg,
                        f'{a} points': gaussians._mask.shape[0],
                        f'{a} visible points': voxel_visible_mask.sum(),
                        f'{a} feat_rate_per_param': F_bits if iteration > opt.triplane_init_fit_iter + 5000 else 0,
                        f'{a} bit_per_feat_param': bit_per_feat_param if iteration > opt.triplane_init_fit_iter else 0,
                        f'{a} bit_per_scaling_param': bit_per_scaling_param if iteration > opt.triplane_init_fit_iter else 0,
                        f'{a} bit_per_offsets_param': bit_per_offsets_param if iteration > opt.triplane_init_fit_iter else 0,
                        f'{a} bit_per_temp_feat_param': bit_per_temp_feat_param if iteration > opt.triplane_init_fit_iter else 0,
                        f'{a} bit_per_param': bit_per_param if iteration > opt.triplane_init_fit_iter else 0,
                    }

                logger.log_metrics(log, step=iteration)

            # ---- report / save (ordering differs between paths, kept separate) ----
            if hyper:
                if iteration in saving_iterations:
                    scene.save(iteration)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    print("Saved checkpoint at iteration {}".format(iteration))
                torch.cuda.synchronize(); t_start_log = time.time()
                training_report(dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, args_param.model_path, mask_weight, args_param, coding_start=opt.triplane_init_fit_iter, num_down_emb_c=dataset.min_embeddings)
                torch.cuda.synchronize(); t_end_log = time.time()
                log_time_sub += t_end_log - t_start_log
            else:
                torch.cuda.synchronize(); t_start_log = time.time()
                training_report(dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, args_param.model_path, mask, args_param, coding_start=opt.triplane_init_fit_iter, num_down_emb_c=dataset.min_embeddings)
                torch.cuda.synchronize(); t_end_log = time.time()
                log_time_sub += t_end_log - t_start_log
                if iteration in saving_iterations:
                    scene.save(iteration)
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    print("Saved checkpoint at iteration {}".format(iteration))

            # ---- densification (#8: different training_statis signature + adjust_anchor_time) ----
            if iteration < opt.update_until and iteration > opt.start_stat:
                if hyper:
                    if iteration not in range(opt.triplane_init_fit_iter + 4999, opt.triplane_init_fit_iter + 7000):
                        for viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, cur_time in zip(viewspace_point_tensor_list, opacity_list, visibility_filter_list, offset_selection_mask_list, voxel_visible_mask_list, time_list):
                            gaussians.training_statis(viewspace_point_tensor, opacity, update_filter=visibility_filter, offset_selection_mask=offset_selection_mask, anchor_visible_mask=voxel_visible_mask, cur_time=cur_time)
                    if iteration >= opt.update_from and iteration % opt.update_interval == 0:
                        if iteration not in range(opt.triplane_init_fit_iter + 5000, opt.triplane_init_fit_iter + 7000) and iteration % args_param.temp_grow_iter != 0:
                            gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                        elif iteration % args_param.temp_grow_iter == 0 and iteration in range(20000, 40000):
                            gaussians.adjust_anchor_time(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                else:
                    if iteration not in range(opt.triplane_init_fit_iter + 4999, opt.triplane_init_fit_iter + 7000):
                        gaussians.training_statis(viewspace_point_tensor, opacity, scaling_for_stats, visibility_filter, offset_selection_mask, voxel_visible_mask, cur_time=viewpoint_cam.time, iteration=iteration)
                    if iteration >= opt.update_from and iteration % opt.update_interval == 0:
                        if iteration not in range(opt.triplane_init_fit_iter + 5000, opt.triplane_init_fit_iter + 7000):
                            gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, iteration=iteration)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # ---- optimizer step (#9: staged / ic_optimizer differs) ----
            if iteration < opt.iterations:
                if hyper:
                    gaussians.optimizer.step()
                    if iteration == 80001:
                        set_requires_grad(gaussians._mask, False)
                    if iteration >= opt.triplane_init_fit_iter:
                        gaussians.feature_net_optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if iteration >= opt.triplane_init_fit_iter:
                        gaussians.feature_net_optimizer.zero_grad(set_to_none=True)
                else:
                    if use_ic_module and iteration > opt.update_from:
                        gaussians.ic_optimizer.step()
                    if iteration < opt.triplane_init_fit_iter:
                        gaussians.optimizer.step()
                    elif iteration >= opt.triplane_init_fit_iter and iteration < 100001:
                        if iteration == opt.triplane_init_fit_iter:
                            set_requires_grad(gaussians.feature_net, True)
                            set_requires_grad(gaussians._mask, False)
                        gaussians.optimizer.step()
                        gaussians.feature_net_optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    if use_ic_module and iteration > opt.update_from:
                        gaussians.ic_optimizer.zero_grad(set_to_none=True)
                    if iteration >= opt.triplane_init_fit_iter:
                        gaussians.feature_net_optimizer.zero_grad(set_to_none=True)

            # ---- mask update strategy (#10: n3dv periodic accumulation, none for hyper) ----
            if not hyper:
                if iteration >= 60000 and iteration < opt.triplane_init_fit_iter:
                    if iteration == 60000:
                        set_requires_grad(gaussians._mask, True)
                    if gaussians._mask.grad is not None:
                        mask_grad_accumulator.append(gaussians._mask.grad.clone())
                    if iteration % mask_update_frequency == 0 and mask_grad_accumulator:
                        avg_grad = torch.stack(mask_grad_accumulator).mean(dim=0)
                        gaussians._mask.grad = avg_grad
                        gaussians.mask_optimizer.step()
                        gaussians.mask_optimizer.zero_grad()
                        mask_grad_accumulator = []
                    else:
                        if gaussians._mask.grad is not None:
                            gaussians._mask.grad = None

    torch.cuda.synchronize(); t_end = time.time()
    return gaussians.x_bound_min, gaussians.x_bound_max


def set_requires_grad(module_or_param, value):
    if isinstance(module_or_param, torch.nn.Module):
        for param in module_or_param.parameters():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                param.requires_grad = value
    elif isinstance(module_or_param, torch.nn.Parameter):
        if module_or_param.dtype.is_floating_point or module_or_param.dtype.is_complex:
            module_or_param.requires_grad = value
    else:
        raise TypeError("Input must be an nn.Module or nn.Parameter")
        
def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, pre_path_name='', weighted_mask=None, args_param=None, coding_start=15000 ,num_down_emb_c=5):
    hyper = (getattr(args_param, 'loader', '') == 'nerfies')

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        rate_set=None
        if 1:
            if iteration >= coding_start or iteration == testing_iterations[-1]:
                with torch.no_grad():
                    log_info, rate_set, total_size = scene.gaussians.estimate_final_bits(None, iteration)
                    print(log_info)
            if iteration >= coding_start and iteration == testing_iterations[-1]:#iteration >= coding_start or iteration == testing_iterations[-1]
                if run_codec:  # conduct encoding and decoding
                    with torch.no_grad():
                        bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
                        # conduct encoding
                        patched_infos, log_info, rate_set, tri_coder_list = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path, weighted_mask=weighted_mask)
                        print(log_info)
                        # conduct decoding
                        log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path, patched_infos=patched_infos, tri_coder_list=tri_coder_list, weighted_mask=weighted_mask)
                        weighted_mask = None

                        print(log_info)
            torch.cuda.empty_cache()
            validation_configs = (
                                  {'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : scene.getValTrainCameras()}
                                  )

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0

                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []
                    psnr_list = []

                    t_list = []


                    for idx, viewpoint in tqdm(enumerate(config['cameras']), ncols=130, desc=f"Evaluating {config['name']}", total=len(config['cameras'])):
                        torch.cuda.synchronize(); t_start = time.time()
                        encoded_temp_feature = scene.gaussians._temperal_anchor_feat[:, 0, :].squeeze()
                        binary_temp_masks = scene.gaussians.get_temp_mask()[:, 0, :]
                        feat_context = None
                        if iteration > 80000 and not scene.gaussians.decoded_version:  # decoded: no re-quantization
                            Q_temp_feat = 1
                            feat_context, feat_rate,f1rate,f2rate,f3rate = scene.gaussians.feature_net(scene.gaussians.get_anchor, itr=iteration)
                            means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                            torch.split(feat_context, split_size_or_sections=[scene.gaussians.feat_dim, scene.gaussians.feat_dim, 6, 6, 3*scene.gaussians.n_offsets, 3*scene.gaussians.n_offsets, scene.gaussians.temp_feat_dim*scene.gaussians.num_temp_feat, scene.gaussians.temp_feat_dim*scene.gaussians.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
                            Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
                            encoded_temp_feature = (STE_multistep.apply(encoded_temp_feature, Q_temp_feat, scene.gaussians._temperal_anchor_feat.mean())).detach()
                        elif iteration > 40000 and not scene.gaussians.decoded_version:
                            Q_temp_feat = 1
                            encoded_temp_feature = (STE_multistep.apply(encoded_temp_feature, Q_temp_feat, scene.gaussians._temperal_anchor_feat.mean())).detach()
                        d_a, d_f, d_o, d_s, _ =  scene.gaussians._deformation(time_emb=viewpoint.time, embeddings=encoded_temp_feature, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_c)
                        d_a = d_a * binary_temp_masks

                        d_f = d_f * binary_temp_masks
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs, step=iteration, d_a=d_a)
                        render_output = renderFunc(viewpoint, scene.gaussians, d_a ,d_f, *renderArgs, visible_mask=voxel_visible_mask, step=iteration, mask=weighted_mask, coding_start=coding_start, test_idx=idx, model_path=scene.model_path, feat_context=feat_context)
                        image = torch.clamp(render_output["render"], 0.0, 1.0)
                        time_sub = render_output["time_sub"]
                        torch.cuda.synchronize(); t_end = time.time()

                        t_list.append(t_end - t_start - time_sub)

                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        gt_image_list.append(gt_image)
                        render_image_list.append(image)
                        errormap_list.append((image - gt_image).abs())

                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_ = psnr(image, gt_image).mean().double()
                        psnr_test += psnr_
                        psnr_list.append(psnr_)
                        ssim_test += 0
                        lpips_test += 0

                    psnr_test /= len(config['cameras'])
                    ssim_test /= len(config['cameras'])
                    lpips_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])

                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                    if config['name'] == 'test':
                        if hyper:
                            for i in range(10):
                                print(f"PSNR {i*30} ~ {(i+1)*30}: {sum(psnr_list[30*i:30*(i+1)])/30}")
                        else:
                            for i in range(6):
                                print(f"PSNR {i*50} ~ {(i+1)*50}: {sum(psnr_list[50*i:50*(i+1)])/50}")

                    if config['name'] == 'test':
                        for image_list, name in zip(
                                [
                                    # gt_image_list,
                                    render_image_list,
                                    # errormap_list,
                                ],
                                [
                                    # "gt",
                                    "ours",
                                    # "error",
                                ]):
                            os.makedirs(os.path.join(args_param.model_path, f"iteration_{iteration}/{config['name']}/{name}"), exist_ok=True)
                            for idx, (img, psnr_) in enumerate(zip(image_list, psnr_list)):
                                torchvision.utils.save_image(img, os.path.join(args_param.model_path, f"iteration_{iteration}/{config['name']}/{name}/{idx:05d}_{psnr_:2.2f}.png"))

                    mask_anchor = scene.gaussians.get_mask_anchor()
                    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()
                    num_of_anchor = scene.gaussians.get_anchor.shape[0]
                    test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                    print(f'Test FPS: {test_fps.item():.5f}')
                    write_to_file_no_compression_ver(args_param.exp_name, config['name'], iteration, psnr_test, ssim_test, lpips_test, args_param, num_of_anchor, mask_anchor_rate)
                    if args_param.exp_name is not None and config['name'] == 'test' and rate_set != None:
                        write_to_file(args_param.exp_name, args_param.lmbda , total_size, psnr_test, ssim_test, lpips_test, rate_set, args_param,num_of_anchor, mask_anchor_rate)


        torch.cuda.empty_cache()

        scene.gaussians.train()


def get_time_mean_var(iteration, scene : Scene, renderFunc, renderArgs, weighted_mask=None, args_param=None, coding_start=15000, num_down_emb_c=None, lambda_dssim=0):
    if getattr(args_param, 'loader', '') == 'nerfies':
   
        scene.gaussians.train()

        scene.gaussians.anchor_start = torch.ones((scene.gaussians.get_anchor.shape[0], 1), device="cuda") * 999
        scene.gaussians.anchor_end = torch.zeros((scene.gaussians.get_anchor.shape[0], 1), device="cuda") - 1
   
        validation_configs = [{'name': 'train', 'cameras' : scene.getTrainCameras()}]

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
            

                with tqdm(enumerate(config['cameras']), ncols=130, desc=f"Evaluating {config['name']}", total=len(config['cameras'])) as pbar:
                    for idx, viewpoint in pbar:
                        torch.cuda.synchronize(); t_start = time.time()
                        encoded_temp_feature = scene.gaussians._temperal_anchor_feat[:, 0, :].squeeze()
                        binary_temp_masks = scene.gaussians.get_temp_mask()[:, 0, :]
                        if iteration > 40000:
                            Q_temp_feat = 1
                            encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat * 1.1
                        d_a, d_f, d_o, d_s, _ =  scene.gaussians._deformation(time_emb=viewpoint.time, embeddings=encoded_temp_feature, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_c)
                        d_a = d_a * binary_temp_masks
                    
                        d_f = d_f * binary_temp_masks
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs, step=iteration, d_a=d_a)
                        scene.gaussians.get_start_end(voxel_visible_mask, idx)

            with torch.no_grad():
                scene.gaussians._temp_mask[:, 1].copy_((scene.gaussians.anchor_start - 1) / len(scene.getTrainCameras()))
                scene.gaussians._temp_mask[:, 2].copy_((scene.gaussians.anchor_end + 1) / len(scene.getTrainCameras()))
                scene.gaussians._temp_mask[:, 3].fill_(math.log(0.1))
                scene.gaussians._temp_mask[:, 4].fill_(math.log(0.1))
    else:
   
        scene.gaussians.train()

        scene.gaussians.anchor_start = torch.ones((scene.gaussians.get_anchor.shape[0], 1), device="cuda") * 999
        scene.gaussians.anchor_end = torch.zeros((scene.gaussians.get_anchor.shape[0], 1), device="cuda") - 1
   
        viewpoint_stack = scene.getTrainCameras(data_loader=True)
        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=args_param.batch_size,shuffle=True,num_workers=16,collate_fn=list)
        loader = iter(viewpoint_stack_loader)
            

        for idx, viewpoint in enumerate(tqdm(loader,
                                         desc=f"[Iter {iteration}] Computing time mean/var",
                                         total=len(viewpoint_stack),
                                         ncols=120)):
            viewpoint = viewpoint[0]
            torch.cuda.synchronize(); t_start = time.time()
            encoded_temp_feature = scene.gaussians._temperal_anchor_feat[:, 0, :].squeeze()
            binary_temp_masks = scene.gaussians.get_temp_mask()[:, 0, :]
            if iteration > 40000:
                Q_temp_feat = 1
                encoded_temp_feature = encoded_temp_feature + torch.empty_like(encoded_temp_feature).uniform_(-0.5, 0.5) * Q_temp_feat * 1.1
            d_a, d_f, d_o, d_s, _ =  scene.gaussians._deformation(time_emb=viewpoint.time, embeddings=encoded_temp_feature, iter=iteration, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_c)
            d_a = d_a * binary_temp_masks
        
            d_f = d_f * binary_temp_masks
            voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs, step=iteration, d_a=d_a)
            scene.gaussians.get_start_end(voxel_visible_mask, viewpoint.time)

        with torch.no_grad():
            scene.gaussians._temp_mask[:, 1].copy_((scene.gaussians.anchor_start - 1/300))
            scene.gaussians._temp_mask[:, 2].copy_((scene.gaussians.anchor_end + 1/300))
            scene.gaussians._temp_mask[:, 3].fill_(math.log(1 / 300))
            scene.gaussians._temp_mask[:, 4].fill_(math.log(1 / 300))


def save_static(dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, pre_path_name='', weighted_mask=None, args_param=None):
   
    if 1:
    
        scene.gaussians.eval()
        if 1:
            viewpoint_cams = scene.getOneTestCamera()

            if 1:
                if 1:
                    l1_test = 0.0
                    psnr_test = 0.0
                    ssim_test = 0.0
                    lpips_test = 0.0

                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []
                    psnr_list = []

                    t_list = []
                    

                    for idx, viewpoint in tqdm(enumerate(viewpoint_cams), ncols=130, desc=f"save statics", total=len(viewpoint_cams)):
                        torch.cuda.synchronize(); t_start = time.time()
                        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs, step=2000)
                        render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask, step=2000, mask=weighted_mask)
                        image = torch.clamp(render_output["render"], 0.0, 1.0)
                        time_sub = render_output["time_sub"]
                        torch.cuda.synchronize(); t_end = time.time()
                        t_list.append(t_end - t_start - time_sub)

                        
                        render_image_list.append(image)
                        

                    
                    for image_list, name in zip([render_image_list], ["static"]):
                        os.makedirs(os.path.join(args_param.model_path, f"static/iteration_{iteration}/{name}"), exist_ok=True)
                        for idx, img in enumerate(image_list):
                            torchvision.utils.save_image(img, os.path.join(args_param.model_path, f"static/iteration_{iteration}/{name}/{idx:05d}.png"))
                    
        torch.cuda.empty_cache()

        scene.gaussians.train()


def write_to_file(txt_name, lmbda, size, psnr, ssim, lpips, rate_set, args_param, num_of_anchor=None, mask_anchor_rate=None):
    # Unified (hyper + n3dv): relative output/<project_name>/<name>_comp.txt, header check/rewrite,
    # trailing num_of_anchor / mask_anchor_rate columns.
    if txt_name is not None:
        txt_name = txt_name + "_comp"
        file_name = f"{txt_name}.txt"
        # Define the column headers and format
        headers = ["lmbda", "size", "psnr", "ssim", "lpips", "anchor", "total_feat"] + [f'feat{i}' for i in range(len(rate_set)-11)] + ["scaling", "offsets", "temp_feat", "temp_mean", "temp_var", "temp_masks", "masks", "MLPs", "hyperprior", "num of anchor", "mask_anchor_rate"]
        header_format = "{:<10} {:<10} {:<10} {:<10} {:<10} " + " ".join(["{:<10}"] * len(rate_set)) + " {:<10} {:<10}" + "\n"
        row_format = "{:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} " + " ".join(["{:<10.4f}"] * len(rate_set)) + "{:<10.4f} {:<10.4f}" + "\n"

        # Define the file directory and check if it exists, create if not
        file_dir = os.path.join(os.getcwd(), f"output/{args_param.project_name}")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, file_name)

        # Check if the file exists and if the header matches
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                header = f.readline()
                if header != header_format.format(*headers):
                    # If header does not match, rewrite the file with the new header
                    with open(file_path, 'w') as f:
                        f.write(header_format.format(*headers))
        else:
            # If file does not exist, write the new header
            with open(file_path, 'w') as f:
                f.write(header_format.format(*headers))

        # Append the new data row to the file
        with open(file_path, 'a') as f:
            f.write(row_format.format(lmbda, size, psnr, ssim, lpips, *rate_set, num_of_anchor, mask_anchor_rate))

        print(f"Data written to {file_path}")

def write_to_file_no_compression_ver(txt_name, mode, iteration, psnr, ssim, lpips, args_param, num_of_anchor=None, mask_anchor_rate=None):
    # Unified (hyper + n3dv): relative output/<project_name>/<name>.txt, header check/rewrite,
    # num_of_anchor / mask_anchor_rate columns.
    if txt_name is not None:
        file_name = f"{txt_name}.txt"
        headers = ['setting', 'mode', 'iteration', 'psnr', 'ssim', 'lpips', 'num of anchor', 'mask_anchor_rate']
        header_format = "{:<30} {:<10} {:<10} {:<10} {:<10} {:<10}  {:<10} {:<10}\n"
        row_format = "{:<30} {:<10} {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}\n"

        # Define the file directory and check if it exists, create if not
        file_dir = os.path.join(os.getcwd(), f"output/{args_param.project_name}")
        file_path = os.path.join(file_dir, file_name)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
            with open(file_path, 'w') as f:
                f.write(header_format.format(*headers))
        else:
            if os.path.exists(file_path):
                # check whether the header matches
                with open(file_path, 'r') as f:
                    header = f.readline()
                    if header != header_format.format(*headers):
                        with open(file_path, 'w') as f:
                            f.write(header_format.format(*headers))
            else:
                with open(file_path, 'w') as f:
                    f.write(header_format.format(*headers))

        # Append the new data row to the file
        with open(file_path, 'a') as f:
            setting = args_param.exp_name
            f.write(row_format.format(setting, mode, iteration, psnr, ssim, lpips, num_of_anchor, mask_anchor_rate))

        print(f"Data written to {file_path}")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    psnr_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background, step=iteration)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        #
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    print('testing_float_psnr=:', sum(psnr_list) / len(psnr_list))

    return t_list, visible_count_list


def render_sets(args_param, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, x_bound_min=None, x_bound_max=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            decoded_version=run_codec,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()
        if x_bound_min is not None:
            gaussians.x_bound_min = x_bound_min
            gaussians.x_bound_max = x_bound_max

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            print(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            print(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000, 40000,50000,60000,70000,80000,90000,  90001,100000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000, 30000, 40000,50000,60000,80000,90000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20000, 30000, 40000,50000,60000, 80000,90000])
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--project_name", type=str, default = 'S3DGS')
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--chcm_slices_list", nargs="+", type=int, default=[5 ,10, 15 ,20]) # chcm 5 10 15 20
    parser.add_argument('--temp_grow_iter', type=int, default=1000)
    parser.add_argument('--temp_loss', action='store_true', default=False)
    parser.add_argument("--chcm_for_offsets", action='store_true', default=False)
    parser.add_argument("--chcm_for_scaling", action='store_true', default=False)
    parser.add_argument("--lmbda", type=float)
    parser.add_argument('--exp_name',  type=str, required=True)
    parser.add_argument('--cam_mask', type=float, default=1)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--lmbda_mask', type=float, default=0.0002)
    parser.add_argument('--use_ic_module', action='store_true', default=False)  # per-view ICModule image enhancement — n3dv opt-in, hyper forced off. Bridged onto dataset in training() so it lands in cfg_args.
    

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations = args.checkpoint_iterations 
    args.save_iterations.append(args.iterations)
    if args.config is not None:
        def merge_hparams(args, config):
            params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
            for param in params:
                if param in config.keys():
                    for key, value in config[param].items():
                        if hasattr(args, key):
                            setattr(args, key, value)

            return args
        import mmengine
        config = mmengine.Config.fromfile(args.config)
        args = merge_hparams(args, config)



    # enable logging
    # Comet logging is optional: fill in your own api_key / workspace below to enable it,
    # or run with --no_comet.
    personal = {'api_key': None, 'workspace': None}
    comet_logger = Experiment(
        api_key=personal['api_key'],
        workspace=personal['workspace'],
        project_name=args.project_name,
        disabled=args.no_comet
    )

    comet_logger.set_name(args.exp_name if getattr(args,'loader','')=='nerfies' else args.exp_name+str(args.lmbda))
    comet_logger.log_parameters(args)
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    print(f'args: {args}')
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    args.port = np.random.randint(10000, 20000)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    x_bound_min, x_bound_max = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from, wandb, logger=comet_logger, load_iteration=args.load_iteration)
    if args.warmup:
        print("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        x_bound_min, x_bound_max = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=comet_logger, load_iteration=args.load_iteration)
    # All done
    print("\nTraining complete.")


