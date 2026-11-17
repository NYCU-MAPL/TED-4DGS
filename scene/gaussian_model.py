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

import os
import time
from functools import reduce
import math

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max, scatter_min
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized
from scene.ic_modules import ICModule
from scene.attribute import attribute_network
from scene.deformation import deform_network


from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    anchor_round_digits, Q_anchor, \
    encoder_anchor, decoder_anchor, \
    encoder, decoder, \
    encoder_gaussian, decoder_gaussian, \
    get_binary_vxl_size

from scene.hexplane import *
import time
import subprocess
from torch import Tensor
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from scene.bitstream.encode import encode, RangeCoder, get_ac_max_val_latent, write_header
from scene.bitstream.decode import decode, compute_offset, fast_get_neighbor
import os
from scene.arm import ArmMLP, get_mu_scale, get_neighbor
import numpy as np
from scene.regulation import compute_plane_smoothness
import matplotlib.pyplot as plt 

bit2MB_scale = 8 * 1024 * 1024
Q_EXP_SCALE = 2**4

def time_bigaussian(mean, var, t, training=False, var_min=0.00001):
    """
    mean: Tensor[...,2], mean[...,0]=a_s, mean[...,1]=a_f
    var:  Tensor[...,2], var[...,0]=log(b_s), var[...,1]=log(b_f)
    t:    Tensor broadcastable to the shape of mean[...,0]
    """
    # split out a_s, a_f
    a_s = mean[..., 0]
    a_f = mean[..., 1]
    # guard: if a_f < a_s, force a_f = a_s
    a_f = torch.maximum(a_f, a_s)

    b_s = var[..., 0].exp() + var_min/2.36
    b_f = var[..., 1].exp() + var_min/2.36

    # m = t-a_s (left) / t-a_f (right) / 0 in the middle interval
    m = torch.where(
        t < a_s,
        t - a_s,
        torch.where(t > a_f, t - a_f, torch.zeros_like(a_s))
    )
    # denom is piecewise too: left b_s / right b_f / middle 1
    denom = torch.where(
        t <  a_s, b_s,
        torch.where(t >  a_f, b_f, torch.ones_like(a_s))
    )
    # final exp(-(m/denom)^2); in the middle m=0 => exp(0)=1
    return torch.exp(- (m / denom) ** 2)
    
class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 chcm_slices_list = [25, 25],
                 chcm_for_offsets = False,
                 chcm_for_scaling = False,
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 attribute_config: dict=None,
                 args=None,
                 comp_start=80001,
                 ):
        self.hyper_mode = (getattr(args, 'loader', '') == 'nerfies')  # single switch: hyper(DSGC) vs n3dv(baseline) behavior
        self.use_ic_module = bool(getattr(args, 'use_ic_module', False)) and not self.hyper_mode  # ICModule: n3dv opt-in, hyper forced off
        if self.hyper_mode:
            super().__init__()
        

            feat_dim = feat_dim
            self.temp_feat_dim = args.anchor_embedding_dim
            self.feat_dim = feat_dim
            self.n_offsets = n_offsets
            self.k_frames_for_one_feature = args.k_frames_for_one_feature
            self.voxel_size = voxel_size
            self.update_depth = update_depth
            self.update_init_factor = update_init_factor
            self.update_hierachy_factor = update_hierachy_factor
            self.use_feat_bank = use_feat_bank
            self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
            self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
            self.n_features_per_level = n_features_per_level
            self.log2_hashmap_size = log2_hashmap_size
            self.log2_hashmap_size_2D = log2_hashmap_size_2D
            self.resolutions_list = resolutions_list
            self.resolutions_list_2D = resolutions_list_2D
            self.ste_binary = ste_binary
            self.ste_multistep = ste_multistep
            self.add_noise = add_noise
            self.Q = Q
            self.use_2D = use_2D
            self.decoded_version = decoded_version
            self.num_frame = 0
        
            self._anchor = torch.empty(0)
            self._offset = torch.empty(0)
            self._mask = torch.empty(0)
            self._temp_mask = torch.empty(0)
            self._anchor_feat = torch.empty(0)
            self._temperal_anchor_feat = torch.empty(0)
            self.num_temp_feat = 1
            self.max_time = 1.0

            self._global_temp_feat = torch.empty(0)

            self._deformation = deform_network(W=args.net_width, D=args.defor_depth, 
                                               min_embeddings=args.min_embeddings, max_embeddings=args.max_embeddings, 
                                               num_frames=args.total_num_frames,
                                               args=args)

            self.opacity_accum = torch.empty(0)

            self._scaling = torch.empty(0)
            self._rotation = torch.empty(0)
            self._opacity = torch.empty(0)
            self.max_radii2D = torch.empty(0)

            self.offset_gradient_accum = torch.empty(0)
            self.offset_denom = torch.empty(0)
            self.significance_num   = torch.empty(0)
            self.significance_denom = torch.empty(0)

            self.anchor_demon = torch.empty(0)

            self.optimizer = None
            self.percent_dense = 0
            self.spatial_lr_scale = 0
            self.setup_functions()
        

            if self.use_feat_bank:
                self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(3+1, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

            mlp_input_feat_dim = feat_dim

            self.mlp_opacity = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1+3, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Tanh()
            ).cuda()

            self.mlp_cov = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1+3, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*self.n_offsets),
            ).cuda()

            self.mlp_color = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1+3, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()

        
            self.mlp_context_from_f1 = nn.Sequential(
                nn.Linear(feat_dim//2, 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, feat_dim),
            ).cuda()

            zero_linear = nn.Linear(2*feat_dim, 3*n_offsets)
            torch.nn.init.zeros_(zero_linear.weight)
            torch.nn.init.zeros_(zero_linear.bias)
            self.mlp_temperal_motion = nn.Sequential(
                nn.Linear(7*(self.temp_feat_dim), 2*feat_dim),
                nn.ReLU(True),
                zero_linear,
            ).cuda()

            self.mlp_temperal_opacity = nn.Sequential(
                nn.Linear(7*(self.temp_feat_dim), 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, n_offsets),
                nn.Sigmoid()
            ).cuda()

            self.mlp_temperal_encoding = nn.Sequential(
                nn.Linear(mlp_input_feat_dim, args.anchor_embedding_dim)
            ).cuda()

            self.mlp_chcm_list = nn.ModuleList()
            self.num_feat_slices = len(chcm_slices_list)
            self.chcm_slices_list = chcm_slices_list
            input_dim = 0
            for i in range(self.num_feat_slices-1):
                input_dim += chcm_slices_list[i]
                self.mlp_chcm_list.append(nn.Sequential(
                    nn.Linear(input_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 2*chcm_slices_list[i+1]),
                ).cuda())

            self.chcm_for_offsets = chcm_for_offsets
            if chcm_for_offsets:
                self.mlp_chcm_offsets = nn.Sequential(
                    nn.Linear(self.feat_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 6*self.n_offsets),
                ).cuda()
            self.chcm_for_scaling = chcm_for_scaling
            if chcm_for_scaling:
                self.mlp_chcm_scaling = nn.Sequential(
                    nn.Linear(self.feat_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 12),
                ).cuda()

            self.attribute_config=attribute_config
            self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        else:
            super().__init__()
            print('hash_params:', use_2D, n_features_per_level,
                  log2_hashmap_size, resolutions_list,
                  log2_hashmap_size_2D, resolutions_list_2D,
                  ste_binary, ste_multistep, add_noise)

            feat_dim = feat_dim
            self.temp_feat_dim = args.anchor_embedding_dim
            self.feat_dim = feat_dim
            self.n_offsets = n_offsets
            self.k_frames_for_one_feature = args.k_frames_for_one_feature
            self.voxel_size = voxel_size
            self.update_depth = update_depth
            self.update_init_factor = update_init_factor
            self.update_hierachy_factor = update_hierachy_factor
            self.use_feat_bank = use_feat_bank
            self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
            self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
            self.n_features_per_level = n_features_per_level
            self.log2_hashmap_size = log2_hashmap_size
            self.log2_hashmap_size_2D = log2_hashmap_size_2D
            self.resolutions_list = resolutions_list
            self.resolutions_list_2D = resolutions_list_2D
            self.ste_binary = ste_binary
            self.ste_multistep = ste_multistep
            self.add_noise = add_noise
            self.Q = Q
            self.use_2D = use_2D
            self.decoded_version = decoded_version
            self.num_frame = 0
            self.comp_start = comp_start
            self._anchor = torch.empty(0)
            self._offset = torch.empty(0)
            self._mask = torch.empty(0)
            self._temp_mask = torch.empty(0)
            self._anchor_feat = torch.empty(0)
            self._temperal_anchor_feat = torch.empty(0)
            self.num_temp_feat = 1
            self.max_time = 1.0

            self._global_temp_feat = torch.empty(0)

            self._deformation = deform_network(W=args.net_width, D=args.defor_depth, 
                                               min_embeddings=args.min_embeddings, max_embeddings=args.max_embeddings, 
                                               num_frames=args.total_num_frames,
                                               args=args)

            self.opacity_accum = torch.empty(0)

            self._scaling = torch.empty(0)
            self._rotation = torch.empty(0)
            self._opacity = torch.empty(0)
            self.max_radii2D = torch.empty(0)

            self.offset_gradient_accum = torch.empty(0)
            self.offset_denom = torch.empty(0)

            self.anchor_demon = torch.empty(0)
            self.significance_num   = torch.empty(0)
            self.significance_denom = torch.empty(0)
            self.optimizer = None
            self.percent_dense = 0
            self.spatial_lr_scale = 0
            self.setup_functions()
            self.anchor_start = torch.empty(0)
            self.anchor_end = torch.empty(0)




            if self.use_feat_bank:
                self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(3+1, feat_dim),
                    nn.ReLU(True),
                    nn.Linear(feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

            mlp_input_feat_dim = feat_dim

            self.mlp_opacity = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, n_offsets),
                nn.Tanh()
            ).cuda()

            self.mlp_cov = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 7*self.n_offsets),
            ).cuda()

            self.mlp_color = nn.Sequential(
                nn.Linear(mlp_input_feat_dim+3+1+3, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()


            zero_linear = nn.Linear(2*feat_dim, 3*n_offsets)
            torch.nn.init.zeros_(zero_linear.weight)
            torch.nn.init.zeros_(zero_linear.bias)
            self.mlp_temperal_motion = nn.Sequential(
                nn.Linear(7*(self.temp_feat_dim), 2*feat_dim),
                nn.ReLU(True),
                zero_linear,
            ).cuda()

            self.mlp_temperal_opacity = nn.Sequential(
                nn.Linear(7*(self.temp_feat_dim), 2*feat_dim),
                nn.ReLU(True),
                nn.Linear(2*feat_dim, n_offsets),
                nn.Sigmoid()
            ).cuda()
    
            self.mlp_temperal_encoding = nn.Sequential(
                nn.Linear(mlp_input_feat_dim, args.anchor_embedding_dim)
            ).cuda()
        


            self.mlp_chcm_list = nn.ModuleList()
            self.num_feat_slices = len(chcm_slices_list)
            self.chcm_slices_list = chcm_slices_list
            input_dim = 0
            for i in range(self.num_feat_slices-1):
                input_dim += chcm_slices_list[i]
                self.mlp_chcm_list.append(nn.Sequential(
                    nn.Linear(input_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 2*chcm_slices_list[i+1]),
                ).cuda())

            self.chcm_for_offsets = chcm_for_offsets
            if chcm_for_offsets:
                self.mlp_chcm_offsets = nn.Sequential(
                    nn.Linear(self.feat_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 6*self.n_offsets),
                ).cuda()
            self.chcm_for_scaling = chcm_for_scaling
            if chcm_for_scaling:
                self.mlp_chcm_scaling = nn.Sequential(
                    nn.Linear(self.feat_dim, 2*feat_dim),
                    nn.ReLU(True),
                    nn.Linear(2*feat_dim, 12),
                ).cuda()

            self.attribute_config=attribute_config
            self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()


    def get_mlp_size(self, digit=32):
        pmlp_o=self.count_parameters(self.mlp_opacity)
        pmlp_rs=self.count_parameters(self.mlp_cov)
        pmlp_color=self.count_parameters(self.mlp_color)
        pmlp_deform = self._deformation.count_deform_mlp_parameters()
        print("MB of mlp_deform:",f"{pmlp_deform*4/10**6}MB")
        pmlp_fa=sum(p.numel() for p in self.feature_net.get_mlp_parameters())
        pmlp_chcm = 0
        for i in range(self.num_feat_slices-1):
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_list[i])
        if self.chcm_for_offsets:
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_offsets)
        if self.chcm_for_scaling:
            pmlp_chcm+=self.count_parameters(self.mlp_chcm_scaling)
        total_mlp_p=pmlp_o+pmlp_rs+pmlp_color+pmlp_fa+pmlp_chcm+pmlp_deform
        print("# param of all MLP:",f"{total_mlp_p/10**6}M")
        print("Size of all MLP(32bit):",f"{total_mlp_p*4/10**6}MB")  
        mlp_size = total_mlp_p*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.mlp_chcm_list.eval()
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.eval()
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.eval()

        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.chcm_for_offsets:
            self.mlp_chcm_offsets.train()
        if self.chcm_for_scaling:
            self.mlp_chcm_scaling.train()
        self.mlp_chcm_list.train()

        if self.use_feat_bank:
            self.mlp_feature_bank.train()
    def capture(self):
        return (
            self.max_radii2D,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
        self.max_radii2D,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)
        torch.cuda.empty_cache()
    

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    def get_mask(self, weighted_mask=None):
        if self.decoded_version:
            return self._mask
        mask_sig = torch.sigmoid(self._mask)
        if weighted_mask is not None:
            mask_sig = mask_sig * weighted_mask
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    def get_temp_mask(self, weighted_mask=None):
        if self.decoded_version:
            return self._temp_mask
        mask_sig = torch.sigmoid(self._temp_mask)
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    
    def get_mask_anchor(self, weighted_mask=None):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._mask)
            if weighted_mask is not None:
                mask_sig = mask_sig * weighted_mask
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    def get_temp_mask_anchor(self, weighted_mask=None):
        with torch.no_grad():
            if self.decoded_version:
                mask_anchor = (torch.sum(self._temp_mask, dim=1)[:, 0]) > 0
                return mask_anchor
            mask_sig = torch.sigmoid(self._temp_mask)
            mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
            mask_anchor = (torch.sum(mask, dim=1)[:, 0]) > 0
            return mask_anchor

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    

    @property
    def get_context_from_f1(self):
        return self.mlp_context_from_f1

    @property
    def get_temporal_encoding_mlp(self):
        return self.mlp_temperal_encoding

    @property
    def get_motion_mlp(self):
        return self.mlp_temperal_motion

    @property
    def get_temperal_opacity_mlp(self):
        return self.mlp_temperal_opacity

    @property
    def get_chcm_mlp_list(self):
        return self.mlp_chcm_list
    
    @property
    def get_chcm_mlp_offsets(self):
        if self.chcm_for_offsets:
            return self.mlp_chcm_offsets
        else:
            return None
    
    @property
    def get_chcm_mlp_scaling(self):
        if self.chcm_for_scaling:
            return self.mlp_chcm_scaling
        else:
            return None
        
    
    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor, quantized_v = Quantize_anchor.apply(self._anchor, self.x_bound_min, self.x_bound_max)
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max

    @torch.no_grad()
    def set_feature_net(self, w=None):
        _anchor = self.get_anchor
        anchornp=_anchor.detach().cpu().numpy()
    
        x = anchornp[:,0]
        y = anchornp[:,1]
        z = anchornp[:,2]
        xyz = np.vstack([x, y, z]).T
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
        labels = lof.fit_predict(xyz)

        dense_points = torch.tensor(xyz[labels == 1], dtype=torch.float32).cuda()

        mean = torch.mean(dense_points, dim=0)
        centered_points = dense_points - mean
        cov_matrix  = torch.cov(centered_points.T)
        # 2. move to CPU to run eigh
        cov_cpu = cov_matrix.detach().cpu()
        eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(cov_cpu)

        # 3. move back to GPU
        eigenvalues = eigenvalues_cpu.to(cov_matrix.device)
        eigenvectors = eigenvectors_cpu.to(cov_matrix.device)
    
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        rotated_points = torch.matmul(centered_points, sorted_eigenvectors)

        standardized_points = rotated_points / torch.sqrt(eigenvalues[sorted_indices])
    
        w=None
        anchor_num=self.get_anchor.shape[0]
        x=round((anchor_num / 36) ** 0.5)
        self.attribute_config['kplanes_config']['resolution'] = [x,x,x]
        self.feature_net = attribute_network(self.attribute_config, 'sof',4).cuda()
        self.feature_net.attribute_net.grid.set_aabb(standardized_points,self.x_bound_max, self.x_bound_min)
        self.feature_net.attribute_net.grid.set_rotation_matrix(sorted_eigenvectors, mean, torch.sqrt(eigenvalues[sorted_indices]))
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_temporal_existence(self, t, mean, var):
        existence = time_bigaussian(mean, var, t, training=self.mlp_color.training)
        return existence

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, num_frame: int):
        self.spatial_lr_scale = spatial_lr_scale
        print(f'spatial_lr_scale: {self.spatial_lr_scale}')
        self.num_frame = num_frame
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        self.num_temp_feat = 1
        self.num_frame = self.num_temp_feat * self.k_frames_for_one_feature
        self.max_time = math.ceil(num_frame / self.k_frames_for_one_feature) * self.k_frames_for_one_feature / num_frame
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda()
        if not self.hyper_mode:
            # n3dv: init mask low (sigmoid 0.04) and keep it frozen until set_masking() at iter 60001
            masks = inverse_sigmoid(torch.tensor(0.04).cuda()) * masks
        temp_masks = inverse_sigmoid(0.015 * torch.ones((fused_point_cloud.shape[0], self.n_offsets, 1)).float().cuda())
    
        print(f'num_temp_feat: {self.num_temp_feat}')
        temperal_anchor_feat = torch.zeros((fused_point_cloud.shape[0], self.num_temp_feat, self.temp_feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        global_anchor_feat = torch.zeros((21, 64)).float().cuda()

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._temperal_anchor_feat = nn.Parameter(temperal_anchor_feat.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        # hyper: mask trainable from the start; n3dv: frozen until set_masking()
        self._mask = nn.Parameter(masks.requires_grad_(self.hyper_mode))
        self._temp_mask = nn.Parameter(temp_masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._global_temp_feat = nn.Parameter(global_anchor_feat.requires_grad_(True))
        self._deformation = self._deformation.to("cuda")
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    def residual_map_setup(self, height, width, number_of_views, training_args, checkpoint_path=None):

        self.ic_modules = nn.ModuleList([
            ICModule(in_ch=3, c=16, num_blocks=2).cuda() 
            for _ in range(number_of_views)
        ])

        l = []
        for i, ic_module in enumerate(self.ic_modules):
            l.append({
                'params': ic_module.parameters(), 
                'lr': training_args.ic_lr_init, 
                "name": f"ic_module"
            })
        
        self.ic_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.ic_modules.load_state_dict(checkpoint['ic_modules'])
            print("Loaded ic_modules from checkpoint.")


    def training_setup(self, training_args):
        if self.hyper_mode:
            self.percent_dense = training_args.percent_dense

            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

            self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.offset_gradient_max = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.anchor_start = torch.ones((self.get_anchor.shape[0], 1), device="cuda") * 999
            self.anchor_end = torch.zeros((self.get_anchor.shape[0], 1), device="cuda") - 1
            self.significance_num   = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device='cuda')
            self.significance_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device='cuda')
            if self.use_feat_bank:
                l = [
                    {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                    {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                    {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                    {'params': [self._temp_mask], 'lr': training_args.temp_feat_lr_init, "name": "temp_mask"},
                    {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                    {'params': [self._temperal_anchor_feat], 'lr': training_args.temp_feat_lr_init, "name": "temperal_anchor_feat"},

                    {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                    {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                    {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                    {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                ]
            else:
                l = [
                    {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                    {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                    {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                    {'params': [self._temp_mask], 'lr': training_args.temp_feat_lr_init, "name": "temp_mask"},
                    {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                    {'params': [self._temperal_anchor_feat], 'lr': training_args.temp_feat_lr_init, "name": "temperal_anchor_feat"},

                    {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                    {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                    {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                    {'params': self.mlp_temperal_motion.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
                    {'params': self.mlp_temperal_opacity.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},

                    {'params': self._deformation.get_mlp_parameters(), 'lr': training_args.mlp_deform_lr_init * self.spatial_lr_scale, "name": "mlp_deform"},
                    {'params': self.mlp_temperal_encoding.parameters(), 'lr': training_args.mlp_deform_lr_init * self.spatial_lr_scale, "name": "mlp_deform"},

                
                ]

            self.l_param=[]
            for item in l:
                if isinstance(item['params'], list):
                    for param in item['params']:
                        self.l_param.append(param)
                else:
                    self.l_param.append(param) 
        
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
            self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        
            self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.offset_lr_delay_mult,
                                                        max_steps=training_args.offset_lr_max_steps)
            self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.mask_lr_delay_mult,
                                                        max_steps=training_args.mask_lr_max_steps,
                                                        step_sub=training_args.mask_lr_start_steps)

            self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                        lr_final=training_args.mlp_opacity_lr_final,
                                                        lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                        max_steps=training_args.mlp_opacity_lr_max_steps)

            self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

            self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.mlp_color_lr_max_steps)
        
            self.hyper_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.hyper_lr_max_steps,
                                                        step_sub=training_args.hyper_lr_start_steps)
            self.arm_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.arm_lr_max_steps,
                                                        step_sub=training_args.arm_lr_start_steps)
        
            if self.use_feat_bank:
                self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                            lr_final=training_args.mlp_featurebank_lr_final,
                                                            lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                            max_steps=training_args.mlp_featurebank_lr_max_steps)

        
        
            self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.mlp_deform_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                        max_steps=training_args.mlp_deform_lr_max_steps)
        
            self.temp_feat_scheduler_args = get_expon_lr_func(lr_init=training_args.temp_feat_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.temp_feat_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.temp_feat_lr_delay_mult,
                                                        max_steps=training_args.temp_feat_lr_max_steps)

            self.temp_mask_scheduler_args = get_expon_lr_func(lr_init=training_args.temp_mask_lr_init,
                                                        lr_final=training_args.temp_mask_lr_final,
                                                        lr_delay_mult=training_args.temp_mask_lr_delay_mult,
                                                        max_steps=training_args.temp_mask_lr_max_steps)
        else:
            self.percent_dense = training_args.percent_dense

            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

            self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.offset_gradient_max = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

            self.anchor_start = torch.ones((self.get_anchor.shape[0], 1), device="cuda") * 999
            self.anchor_end = torch.zeros((self.get_anchor.shape[0], 1), device="cuda") - 1
            self.significance_num   = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device='cuda')
            self.significance_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device='cuda')

            if self.use_feat_bank:
                l = [
                    {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                    {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                    {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                    {'params': [self._temp_mask], 'lr': training_args.temp_feat_lr_init, "name": "temp_mask"},
                    {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                    {'params': [self._temperal_anchor_feat], 'lr': training_args.temp_feat_lr_init, "name": "temperal_anchor_feat"},

                    {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                    {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                    {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                    {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                
                ]
            else:
                l = [
                    {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                    {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                    {'params': [self._temp_mask], 'lr': training_args.temp_feat_lr_init, "name": "temp_mask"},
                    {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                    {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                    {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                    {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                    {'params': [self._temperal_anchor_feat], 'lr': training_args.temp_feat_lr_init, "name": "temperal_anchor_feat"},

                    {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                    {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                    {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                
                    {'params': self.mlp_temperal_motion.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
                    {'params': self.mlp_temperal_opacity.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},

                    {'params': self._deformation.get_mlp_parameters(), 'lr': training_args.mlp_deform_lr_init * self.spatial_lr_scale, "name": "mlp_deform"},
                    {'params': self.mlp_temperal_encoding.parameters(), 'lr': training_args.mlp_deform_lr_init * self.spatial_lr_scale, "name": "mlp_deform"},


                
                ]

            self.l_param=[]
            for item in l:
                if isinstance(item['params'], list):
                    for param in item['params']:
                        self.l_param.append(param)
                else:
                    self.l_param.append(param) 
        
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
            l = [
                    {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                ]
            self.mask_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
            self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        
            self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.offset_lr_delay_mult,
                                                        max_steps=training_args.offset_lr_max_steps)
            self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.mask_lr_delay_mult,
                                                        max_steps=training_args.mask_lr_max_steps,
                                                        step_sub=training_args.mask_lr_start_steps)

            self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                        lr_final=training_args.mlp_opacity_lr_final,
                                                        lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                        max_steps=training_args.mlp_opacity_lr_max_steps)

            self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

            self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.mlp_color_lr_max_steps)
        
            self.hyper_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.hyper_lr_max_steps,
                                                        step_sub=training_args.hyper_lr_start_steps)
            self.arm_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.arm_lr_max_steps,
                                                        step_sub=training_args.arm_lr_start_steps)
        
            if self.use_feat_bank:
                self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                            lr_final=training_args.mlp_featurebank_lr_final,
                                                            lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                            max_steps=training_args.mlp_featurebank_lr_max_steps)

        
        
            self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.mlp_deform_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                        max_steps=training_args.mlp_deform_lr_max_steps)
        
            self.temp_feat_scheduler_args = get_expon_lr_func(lr_init=training_args.temp_feat_lr_init * self.spatial_lr_scale,
                                                        lr_final=training_args.temp_feat_lr_final * self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.temp_feat_lr_delay_mult,
                                                        max_steps=training_args.temp_feat_lr_max_steps)
        
            self.temp_mask_scheduler_args = get_expon_lr_func(lr_init=training_args.temp_mask_lr_init,
                                                        lr_final=training_args.temp_mask_lr_final,
                                                        lr_delay_mult=training_args.temp_mask_lr_delay_mult,
                                                        max_steps=training_args.temp_mask_lr_max_steps,
                                                        step_sub=training_args.temp_mask_lr_start_steps)

            self.ic_scheduler_args = get_expon_lr_func(lr_init=training_args.ic_lr_init,
                                                        lr_final=training_args.ic_lr_final,
                                                        lr_delay_mult=training_args.ic_lr_delay_mult,
                                                        max_steps=training_args.ic_lr_max_steps,
                                                        step_sub=training_args.ic_lr_start_steps)

            self.inr_hyper_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                        lr_final=training_args.mlp_color_lr_final,
                                                        lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                        max_steps=training_args.inr_hyper_lr_max_steps,
                                                        step_sub=training_args.inr_hyper_lr_start_steps)
        
        
            self.entropy_model_scheduler_args = get_expon_lr_func(lr_init=training_args.entropy_model_lr_init,
                                                        lr_final=training_args.entropy_model_lr_final,
                                                        lr_delay_mult=training_args.entropy_model_lr_delay_mult,
                                                        max_steps=training_args.entropy_model_lr_max_steps,
                                                        step_sub= training_args.entropy_model_lr_step_sub)

            self.hyper_scheduler_args = get_expon_lr_func(lr_init=training_args.hyper_lr_init,
                                                        lr_final=training_args.hyper_lr_final,
                                                        lr_delay_mult=training_args.hyper_lr_delay_mult,
                                                        max_steps=training_args.hyper_lr_max_steps,
                                                        step_sub= training_args.hyper_lr_start_steps)


    def training_setup_triplane(self, training_args):
        
        feature_net_l = [
            {'params': self.feature_net.get_mlp_parameters(), 'lr': training_args.mlp_color_lr_init, "name": "hyper"},
            {'params': self.mlp_chcm_list.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "hyper"},
        ]
        if self.chcm_for_offsets:
            feature_net_l.append({'params': self.mlp_chcm_offsets.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "hyper"})
        if self.chcm_for_scaling:
            feature_net_l.append({'params': self.mlp_chcm_scaling.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "hyper"})
        self.feature_net_optimizer = torch.optim.Adam(feature_net_l, lr=0.0, eps=1e-15)
        

        
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters())
    def count_parameters_l(self,params_list):
        return sum(p.numel() for p in params_list)
    def update_learning_rate(self, iteration):
        if self.hyper_mode:
            ''' Learning rate scheduling per step '''
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "offset":
                    lr = self.offset_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mask":
                    lr = self.mask_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "anchor":
                    lr = self.anchor_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_opacity":
                    lr = self.mlp_opacity_scheduler_args(iteration)
                    param_group['lr'] = lr
                if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                    lr = self.mlp_featurebank_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_cov":
                    lr = self.mlp_cov_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_color":
                    lr = self.mlp_color_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_deform":
                    lr = self.mlp_deform_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "temperal_anchor_feat":
                    lr = self.temp_feat_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "temp_mask":
                    lr = self.temp_mask_scheduler_args(iteration)
                    param_group['lr'] = lr
        else:
            ''' Learning rate scheduling per step '''
            if self.use_ic_module:
                for param_group in self.ic_optimizer.param_groups:
                    if param_group["name"] == "ic_module":
                        if iteration > 60000:
                            iteration2 = iteration - 30000
                            lr = self.ic_scheduler_args(iteration2)
                        elif iteration >= 80000:
                            iteration3 = iteration - 50000
                            lr = self.ic_scheduler_args(iteration3)
                        else:
                            lr = self.ic_scheduler_args(iteration)
                        param_group['lr'] = lr

            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "temp_mask":
                    lr = self.temp_mask_scheduler_args(iteration)
                    param_group['lr'] = lr
                
                if param_group["name"] == "offset":
                    lr = self.offset_scheduler_args(iteration)
                    param_group['lr'] = lr
            
                if param_group["name"] == "anchor":
                    lr = self.anchor_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_opacity":
                    lr = self.mlp_opacity_scheduler_args(iteration)
                    param_group['lr'] = lr
                if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                    lr = self.mlp_featurebank_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_cov":
                    lr = self.mlp_cov_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_color":
                    lr = self.mlp_color_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "mlp_deform":
                    lr = self.mlp_deform_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "temperal_anchor_feat":
                    lr = self.temp_feat_scheduler_args(iteration)
                    param_group['lr'] = lr

            for param_group in self.mask_optimizer.param_groups:
                if param_group["name"] == "mask":
                    lr = self.mask_scheduler_args(iteration)
                    param_group['lr'] = lr  
            


            if  iteration > self.comp_start:
                for param_group in self.feature_net_optimizer.param_groups:
                    if param_group["name"] == "hyper":
                            lr = self.inr_hyper_scheduler_args(iteration)
                            param_group['lr'] = lr

                

                

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._temp_mask.shape[1]*self._temp_mask.shape[2]):
            l.append('f_temp_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._temperal_anchor_feat.shape[1]*self._temperal_anchor_feat.shape[2]):
            l.append('f_temp_anchor_feat_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        temp_mask = self._temp_mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        temp_anchor_feat = self._temperal_anchor_feat.detach().flatten(start_dim=1).cpu().numpy()



        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, mask, temp_mask, anchor_feat, opacities, scale, rotation, temp_anchor_feat), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_only_ply(self, path):
        mask = self.get_mask_anchor()
        anchor = self.get_anchor[mask].detach().cpu().numpy()

        dtype_xyz = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

        # 3. convert (N,3) to a structured array
        elements = np.empty(anchor.shape[0], dtype=dtype_xyz)
        elements['x'] = anchor[:, 0]
        elements['y'] = anchor[:, 1]
        elements['z'] = anchor[:, 2]

        # 4. write out PLY
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
    
    def set_masking(self):
        init_val = 0.04
        with torch.no_grad():
            new_masks = inverse_sigmoid(init_val * torch.ones((self._anchor.shape[0], self.n_offsets, 1), dtype=torch.float32, device="cuda"))
            self._mask.data.copy_(new_masks)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        anchor_temp_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_temp_anchor_feat_")]
        anchor_temp_feat_names = sorted(anchor_temp_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_temp_feat = np.zeros((anchor.shape[0], len(anchor_temp_feat_names)))
        for idx, attr_name in enumerate(anchor_temp_feat_names):
            anchor_temp_feat[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        anchor_temp_feat = anchor_temp_feat.reshape((anchor_temp_feat.shape[0], self.num_temp_feat, -1))
    
        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        temp_mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_temp_mask")]
        temp_mask_names = sorted(temp_mask_names, key = lambda x: int(x.split('_')[-1]))
        temp_masks = np.zeros((anchor.shape[0], len(temp_mask_names)))
        for idx, attr_name in enumerate(temp_mask_names):
            temp_masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        temp_masks = temp_masks.reshape((temp_masks.shape[0], 1, -1))
    

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._temp_mask = nn.Parameter(torch.tensor(temp_masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._temperal_anchor_feat = nn.Parameter(torch.tensor(anchor_temp_feat, dtype=torch.float, device="cuda").requires_grad_(True))

        del plydata
        torch.cuda.empty_cache()
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        if self.hyper_mode:
            optimizable_tensors = {}
            for group in self.optimizer.param_groups:
                if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'hyper' in group['name'] or 'arm' in group['name']:
                    continue
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

            return optimizable_tensors
        else:
            optimizable_tensors = {}
            for group in self.optimizer.param_groups:
                if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'hyper' in group['name'] or 'arm' in group['name']:
                    continue
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

            for group in self.mask_optimizer.param_groups:
                if group["name"] == "mask":
                    assert len(group["params"]) == 1
                    extension_tensor = tensors_dict[group["name"]]
                    stored_state = self.mask_optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                        del self.mask_optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        self.mask_optimizer.state[group['params'][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]

            return optimizable_tensors


    def get_start_end(self, anchor_visible_mask, iter=0):
        self.anchor_start[anchor_visible_mask] = torch.clamp(self.anchor_start[anchor_visible_mask], max=iter)
        self.anchor_end[anchor_visible_mask] = torch.clamp(self.anchor_end[anchor_visible_mask], min=iter)

    def training_statis(self, viewspace_point_tensor, opacity, scaling_for_stats=None, update_filter=None, offset_selection_mask=None, anchor_visible_mask=None, cur_time=None, iteration=0):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    
        self.significance_num[combined_mask]   += grad_norm
        self.significance_denom[combined_mask] += 1

        cur_val = self.offset_gradient_accum[combined_mask]

        update_mask = grad_norm > cur_val
        self.offset_gradient_accum[combined_mask] = torch.max(cur_val, grad_norm)
        self.offset_denom[combined_mask][update_mask] = cur_time
    def _prune_anchor_optimizer(self, mask):
        if self.hyper_mode:
            optimizable_tensors = {}
            for group in self.optimizer.param_groups:
                if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                    continue

                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    if group['name'] == "scaling":
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    if group['name'] == "scaling":
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]


            return optimizable_tensors
        else:
            optimizable_tensors = {}
            for group in self.optimizer.param_groups:
                if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name']:
                    continue

                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state
                    if group['name'] == "scaling":
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    if group['name'] == "scaling":
                        scales = group["params"][0]
                        temp = scales[:,3:]
                        temp[temp>0.05] = 0.05
                        group["params"][0][:,3:] = temp
                    optimizable_tensors[group["name"]] = group["params"][0]

            for group in self.mask_optimizer.param_groups:
                if group["name"] == "mask":
                    stored_state = self.mask_optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                        del self.mask_optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                        self.mask_optimizer.state[group['params'][0]] = stored_state
                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]
            return optimizable_tensors


    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._temp_mask = optimizable_tensors["temp_mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._temperal_anchor_feat = optimizable_tensors["temperal_anchor_feat"]


    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_temperal_feat = self._temperal_anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, self.num_temp_feat, self.temp_feat_dim])[candidate_mask]
                new_temperal_feat = scatter_max(new_temperal_feat, inverse_indices.unsqueeze(1).expand(-1, new_temperal_feat.size(1)), dim=0)[0][remove_duplicates]

                new_temp_masks = self._temp_mask.unsqueeze(dim=1).repeat([1, self.n_offsets, 1, 1]).view([-1, self.n_offsets, 1])[candidate_mask]
                new_temp_masks = scatter_max(new_temp_masks, inverse_indices.unsqueeze(1).expand(-1, new_temp_masks.size(1)), dim=0)[0][remove_duplicates]
                

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_temp_masks = inverse_sigmoid(0.01 * torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda())

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                    "temperal_anchor_feat": new_temperal_feat,
                    "temp_mask": new_temp_masks
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                
                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]
                self._temperal_anchor_feat = optimizable_tensors["temperal_anchor_feat"]
                self._temp_mask = optimizable_tensors["temp_mask"]


    def anchor_growing_time(self, grads, threshold, grad_max_time):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(1):  # 3
            cur_threshold = threshold * 32

            duration_mask = (self._temp_mask[:, 2, :].clamp(max=1.0) - self._temp_mask[:, 1, :].clamp(min=0.0)).repeat([1, self.n_offsets, 1]).view(-1) > 0.1

            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, duration_mask)

            if_dynamic = self.get_temp_mask()[:, 0, :].repeat([1, self.n_offsets, 1]).view([-1, 1]).squeeze()


            candidate_mask = torch.logical_and(candidate_mask, if_dynamic)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
           

            candidate_mask = candidate_mask.view(-1, self.n_offsets).any(dim=1)

            if torch.sum(candidate_mask) > 0:
                new_scaling = self._scaling[candidate_mask]
                new_rotation = self._rotation[candidate_mask]
                candidate_anchor = self._anchor[candidate_mask]
                new_opacities = self._opacity[candidate_mask]
                new_feat = self._anchor_feat[candidate_mask]
                new_temperal_feat = self._temperal_anchor_feat[candidate_mask]
                new_offsets = self._offset[candidate_mask]
                new_masks = self._mask[candidate_mask]
                new_temp_masks = self._temp_mask[candidate_mask]
                grad_max_time = grad_max_time[candidate_mask]
                new_temp_masks_before = new_temp_masks.clone()
                new_temp_masks_before[:, 1, 0] = grad_max_time.view(-1)
                new_temp_masks_after = new_temp_masks.clone()
                new_temp_masks_after[:, 2, 0] = grad_max_time.view(-1)


                d = {
                    "anchor": torch.cat([candidate_anchor, candidate_anchor], dim=0),
                    "scaling": torch.cat([new_scaling, new_scaling], dim=0),
                    "rotation": torch.cat([new_rotation, new_rotation], dim=0),
                    "anchor_feat": torch.cat([new_feat, new_feat], dim=0),
                    "offset": torch.cat([new_offsets, new_offsets], dim=0),
                    "mask": torch.cat([new_masks, new_masks], dim=0),
                    "opacity": torch.cat([new_opacities, new_opacities], dim=0),
                    "temperal_anchor_feat": torch.cat([new_temperal_feat, new_temperal_feat], dim=0),
                    "temp_mask": torch.cat([new_temp_masks_before, new_temp_masks_after], dim=0)
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0]*2, 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0]*2, 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]
                self._temperal_anchor_feat = optimizable_tensors["temperal_anchor_feat"]
                self._temp_mask = optimizable_tensors["temp_mask"]

                # anchors_rounded: (N, 3)
            
            return candidate_mask
    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005, iteration=0):
        # 1) compute each anchor's Temporal Significance
        num   = self.significance_num.view(-1, self.n_offsets, 1)
        denom = self.significance_denom.view(-1, self.n_offsets, 1)
        denom[denom == 0] = 1e-6
        temporal_grad = (num / denom).squeeze(1)  # (num_anchors,)
        grads = self.significance_num / self.significance_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)

        offset_mask = (self.significance_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

    
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        # update offset_gradient_denom
        self.significance_denom[offset_mask] = 0
        padding_significance_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.significance_denom.shape[0], 1],
                                             dtype=torch.float,
                                             device=self.significance_denom.device)
        self.significance_denom = torch.cat([self.significance_denom, padding_significance_denom], dim=0)
    
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # update significance_num
        self.significance_num[offset_mask] = 0
        padding_significance_num = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.significance_num.shape[0], 1],
                                                dtype=torch.float,
                                                device=self.significance_num.device)        
        self.significance_num = torch.cat([self.significance_num, padding_significance_num], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        significance_num = self.significance_num.view([-1, self.n_offsets])[~prune_mask]
        significance_num = significance_num.view([-1, 1])
        del self.significance_num
        self.significance_num = significance_num

        significance_denom = self.significance_denom.view([-1, self.n_offsets])[~prune_mask]
        significance_denom = significance_denom.view([-1, 1])
        del self.significance_denom
        self.significance_denom = significance_denom

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    def adjust_anchor_time(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        num   = self.significance_num.view(-1, self.n_offsets, 1)
        denom = self.significance_denom.view(-1, self.n_offsets, 1)
        denom[denom == 0] = 1e-6
        temporal_grad = (num / denom).squeeze(1)  # (num_anchors,)
        grads = self.offset_gradient_accum
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)

        anchor_grads_norm = grads_norm.view(-1, self.n_offsets)
        anchor_max_time = self.offset_denom.view(-1, self.n_offsets)

        # get max time with max grad per anchor
        max_grad_per_anchor = torch.argmax(anchor_grads_norm, dim=1)
        max_time_per_anchor = anchor_max_time[torch.arange(anchor_max_time.shape[0]), max_grad_per_anchor]

        
        prune_mask = self.anchor_growing_time(grads_norm, grad_threshold, max_time_per_anchor)

        padding_prune_mask = torch.zeros([self.get_anchor.shape[0] - prune_mask.shape[0]], dtype=torch.bool, device='cuda')
        prune_mask = torch.cat([prune_mask, padding_prune_mask], dim=0)

        self.offset_denom = torch.zeros_like(self.offset_denom, device=self.offset_denom.device)
        
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)


        padding_significance_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.significance_denom.shape[0], 1],
                                             dtype=torch.float,
                                             device=self.significance_denom.device)
        self.significance_denom = torch.cat([self.significance_denom, padding_significance_denom], dim=0)
        
        self.offset_gradient_accum = torch.zeros_like(self.offset_gradient_accum, device=self.offset_gradient_accum.device)
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        padding_significance_num = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.significance_num.shape[0], 1],
                                                dtype=torch.float,
                                                device=self.significance_num.device)        
        self.significance_num = torch.cat([self.significance_num, padding_significance_num], dim=0)

        # # prune anchors
        

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        significance_num = self.significance_num.view([-1, self.n_offsets])[~prune_mask]
        significance_num = significance_num.view([-1, 1])
        del self.significance_num
        self.significance_num = significance_num

        significance_denom = self.significance_denom.view([-1, self.n_offsets])[~prune_mask]
        significance_denom = significance_denom.view([-1, 1])
        del self.significance_denom
        self.significance_denom = significance_denom

        

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")




    def save_mlp_checkpoints(self,path):
        if self.hyper_mode:
            mkdir_p(os.path.dirname(path))

        

            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                }, path)
            else:
                print('saving mlp checkpoints')
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'deformation': self._deformation.state_dict(),
                }, path)
        else:
            mkdir_p(os.path.dirname(path))

            checkpoint = {}

            if self.use_feat_bank:
                checkpoint.update({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                })
            else:
                checkpoint.update({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp':     self.mlp_cov.state_dict(),
                    'color_mlp':   self.mlp_color.state_dict(),
                    'deformation': self._deformation.state_dict(),
                    'context_mlp': self.mlp_chcm_list.state_dict(),
                    'feat_base': self.feature_net.state_dict() if hasattr(self, 'feature_net') else None,
                })
                if self.use_ic_module:
                    checkpoint['ic_modules'] = self.ic_modules.state_dict()

            # save entropy_model
            if hasattr(self, 'entropy_model') and self.entropy_model is not None:
                self.entropy_model.update() 
                checkpoint['entropy_model'] = self.entropy_model.state_dict()
        
            # save the main checkpoint
            torch.save(checkpoint, path)
        
    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
    
        self._deformation.load_state_dict(checkpoint['deformation'])
        self._deformation.cuda()

        if checkpoint.get('feat_base') is not None:
            print('loading feature net') 
            self.feature_net.load_state_dict(checkpoint['feat_base'])
            self.feature_net.cuda()

            self.mlp_chcm_list.load_state_dict(checkpoint['context_mlp'])
            self.mlp_chcm_list.cuda()
    
        torch.cuda.empty_cache()
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self, weighted_mask, iteration=0):
        Q_feat = 1
        Q_temp_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2
        Q_temp_mean = 0.1
        Q_temp_var = 0.1

        mask_anchor = self.get_mask_anchor(weighted_mask)
        temp_mask_anchor = self.get_temp_mask(weighted_mask)[:, 0, :].squeeze()
        temp_mask_after_mask_anchor = temp_mask_anchor[mask_anchor]
        temp_mask_after_mask_anchor = temp_mask_after_mask_anchor.bool()
        temp_mask = torch.logical_and(temp_mask_anchor, mask_anchor)
        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _temp_feat = self._temperal_anchor_feat[temp_mask]
        mean_t, var_t = self._temp_mask[:, 1:3, 0][mask_anchor], self._temp_mask[:, 3:5, 0][mask_anchor]
        _mask = self.get_mask(weighted_mask)[mask_anchor]
        feat_context, feat_rate, f1rate, f2rate, f3rate = self.feature_net(_anchor, itr=-1)
        means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, self.temp_feat_dim*self.num_temp_feat, self.temp_feat_dim*self.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
        list_of_means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
        list_of_scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
        Q_feat = Q_feat * (1.1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
        Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
        Q_temp_feat = Q_temp_feat[temp_mask_after_mask_anchor]   
        Q_temp_mean = Q_temp_mean * (1.1 + torch.tanh(Q_temp_mean_adj))
        Q_temp_var = Q_temp_var * (1.1 + torch.tanh(Q_temp_var_adj))


        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        feat_list = list(torch.split(_feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
        bit_feat_list = []
        for i in range(self.num_feat_slices):
            if i == 0:
                bit_feat = self.entropy_gaussian.forward(feat_list[i], list_of_means[i], list_of_scales[i], Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = feat_list[0]
            else:
                dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                mean = list_of_means[i] + dmean
                scale = list_of_scales[i] + dscale
                bit_feat = self.entropy_gaussian.forward(feat_list[i], mean, scale, Q_feat)
                bit_feat_list.append(torch.sum(bit_feat).item())
                decoded_feat = torch.cat([decoded_feat, feat_list[i]], dim=-1)


        if self.chcm_for_scaling:
            dmean_scaling, dscale_scaling = torch.split(self.mlp_chcm_scaling(decoded_feat), split_size_or_sections=[6, 6], dim=-1)
            mean_scaling = mean_scaling + dmean_scaling
            scale_scaling = scale_scaling + dscale_scaling
        if self.chcm_for_offsets:
            dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat), split_size_or_sections=[3*self.n_offsets, 3*self.n_offsets], dim=-1)
            mean_offsets = mean_offsets + dmean_offsets
            scale_offsets = scale_offsets + dscale_offsets
        

        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        temp_feat = (STE_multistep.apply(_temp_feat, Q_temp_feat.unsqueeze(1))).detach()
        temp_feat = temp_feat.view(-1, self.num_temp_feat*self.temp_feat_dim)
        offsets = offsets.view(-1, 3*self.n_offsets)
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        mean_t = (STE_multistep.apply(mean_t, Q_temp_mean)).detach()
        var_t = (STE_multistep.apply(var_t, Q_temp_var)).detach()
        
        bit_temp_mean = self.entropy_gaussian.forward(mean_t, mean_temp_mean, scale_temp_mean, Q_temp_mean)
        bit_temp_var = self.entropy_gaussian.forward(var_t, mean_temp_var, scale_temp_var, Q_temp_var)
        bit_temp_mean = torch.sum(bit_temp_mean).item()
        bit_temp_var = torch.sum(bit_temp_var).item()

        bit_temp_mask = get_binary_vxl_size(temp_mask_after_mask_anchor)[1].item()

        mean_temp_feat = mean_temp_feat[temp_mask_after_mask_anchor]
        scale_temp_feat = scale_temp_feat[temp_mask_after_mask_anchor]
        bit_temp_feat = self.entropy_gaussian.forward(temp_feat, mean_temp_feat, scale_temp_feat, Q_temp_feat)
        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp
        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        bit_temp_feat = torch.sum(bit_temp_feat).item()
        
        bit_masks = get_binary_vxl_size(_mask)[1].item()
        ftrirate=feat_rate / bit2MB_scale
        total = round((bit_anchor + sum(bit_feat_list) + bit_scaling + bit_offsets + bit_masks + bit_temp_feat + bit_temp_var + bit_temp_mask + bit_temp_mean + self.get_mlp_size()[0])/bit2MB_scale+ftrirate, 4)
        log_info = f"\nEstimated sizes in MB: " \
                f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                f"total feat {round(sum(bit_feat_list)/bit2MB_scale, 4)}, " \
                f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                f"temp_feat {round(bit_temp_feat/bit2MB_scale, 4)}, " \
                f"temp_mean {round(bit_temp_mean/bit2MB_scale, 4)}, " \
                f"temp_var {round(bit_temp_var/bit2MB_scale, 4)}, " \
                f"temp_mask {round(bit_temp_mask/bit2MB_scale, 4)}, " \
                f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                f"Triplane_f {round(ftrirate,4)}," \
                f"Total {total}"
        mb_feat_list = [round(bit_feat/bit2MB_scale, 4) for bit_feat in bit_feat_list]
        rate_set=[round(bit_anchor/bit2MB_scale, 4), round(sum(bit_feat_list)/bit2MB_scale, 4)] + mb_feat_list + [round(bit_scaling/bit2MB_scale, 4), round(bit_offsets/bit2MB_scale, 4), round(bit_temp_feat/bit2MB_scale, 4), round(bit_temp_mean/bit2MB_scale, 4), round(bit_temp_var/bit2MB_scale, 4), round(bit_temp_mask/bit2MB_scale, 4), round(bit_masks/bit2MB_scale, 4),round(self.get_mlp_size()[0]/bit2MB_scale, 4), round(ftrirate,4)]
        
        return log_info, rate_set, total

       

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name, weighted_mask):

        t_codec = 0

        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        mask_anchor = self.get_mask_anchor(weighted_mask)
        temp_mask_anchor = self.get_temp_mask(weighted_mask)[:, 0, :].squeeze()

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _temp_feat = self._temperal_anchor_feat[mask_anchor]
        _mask = self.get_mask(weighted_mask)[mask_anchor]
        _temp_mask = self.get_temp_mask(weighted_mask)[mask_anchor][:,0]

        _mean_t, _var_t = self._temp_mask[:, 1:3, 0][mask_anchor], self._temp_mask[:, 3:5, 0][mask_anchor]
        N = _anchor.shape[0]
        MAX_batch_size = 500
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_lists = [[] for _ in range(self.num_feat_slices)]
        
        bit_scaling_list = []
        bit_offsets_list = []
        bit_temp_feat_list = []
        bit_temp_mean_list = []
        bit_temp_var_list = []
        anchor_infos_list = []
        indices_list = []
        min_feat_lists = [[] for _ in range(self.num_feat_slices)]
        max_feat_lists = [[] for _ in range(self.num_feat_slices)]
        
        min_scaling_list = []
        max_scaling_list = []
        min_offsets_list = []
        max_offsets_list = []
        min_temp_feat_list = []
        max_temp_feat_list = []
        min_mean_t_list = []
        max_mean_t_list = []
        min_var_t_list = []
        max_var_t_list = []
 

        feat_lists = [[] for _ in range(self.num_feat_slices)]
        scaling_list = []
        offsets_list = []
        temp_feat_list = []
        mean_t_list = []
        var_t_list = []

        masks_b_name = os.path.join(pre_path_name, 'masks.b')
        temp_masks_b_name = os.path.join(pre_path_name, 'temp_masks.b')

        torch.save(_anchor, os.path.join(pre_path_name, 'anchor.pkl'))

        for s in range(steps):
            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name_list = []
            for i in range(self.num_feat_slices):
                feat_b_name_list.append(os.path.join(pre_path_name, f'feat{i}.b').replace('.b', f'_{s}.b'))
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')
            temp_feat_b_name = os.path.join(pre_path_name, 'temp_feat.b').replace('.b', f'_{s}.b')
            
            mean_t_b_name = os.path.join(pre_path_name, 'mean_t.b').replace('.b', f'_{s}.b')
            var_t_b_name = os.path.join(pre_path_name, 'var_t.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2
            Q_temp_feat = 1

            Q_temp_mean = 0.1
            Q_temp_var = 0.1

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)
            indices_list.append(indices+N_start)

            anchor_sort = _anchor[N_start:N_end][indices]  # [N_num, 3]
            feat_context, feat_rate, f1rate, f2rate, f3rate  = self.feature_net(anchor_sort, itr=-1)
            means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, self.temp_feat_dim*self.num_temp_feat, self.temp_feat_dim*self.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)

            means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
            scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
        
        
            Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in self.chcm_slices_list]
            

            feat = _feat[N_start:N_end][indices]  # [N_num*32]

            _feat_list = list(torch.split(feat, split_size_or_sections=self.chcm_slices_list, dim=-1))
            
            for i in range(self.num_feat_slices):
                if i == 0:
                    temp_feat = _feat_list[i].reshape(-1)
                    temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], self._anchor_feat.mean())  # align STE grid to training-time anchor (was masked-subset mean)                    
                    mean = means[i].reshape(-1)
                    scale = torch.clamp(scales[i].reshape(-1), min=1e-9)
                    bit_feat, min_feat, max_feat = encoder_gaussian(temp_feat, mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i])
                    bit_feat_lists[i].append(bit_feat)
                    min_feat_lists[i].append(min_feat)
                    max_feat_lists[i].append(max_feat)
                    temp_feat = temp_feat.reshape(N_num, -1)
                    feat_lists[i].append(temp_feat)
                    decoded_feat = temp_feat
                    
                else:
                    temp_feat = _feat_list[i].reshape(-1)
                    temp_feat = STE_multistep.apply(temp_feat, Q_feat_list[i], self._anchor_feat.mean())  # align STE grid to training-time anchor (was masked-subset mean)
                    
                    dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                    mean = (means[i] + dmean).reshape(-1)
                    scale = torch.clamp((scales[i] + dscale).reshape(-1), min=1e-9)
                    bit_feat, min_feat, max_feat = encoder_gaussian(temp_feat, mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i])
                    bit_feat_lists[i].append(bit_feat)
                    min_feat_lists[i].append(min_feat)
                    max_feat_lists[i].append(max_feat)
                    temp_feat = temp_feat.reshape(N_num, -1)
                    feat_lists[i].append(temp_feat)
                    decoded_feat = torch.cat([decoded_feat, temp_feat], dim=-1)

            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            Q_temp_feat_adj = Q_temp_feat_adj.contiguous().repeat(1, mean_temp_feat.shape[-1]).view(-1)
            Q_temp_mean_adj = Q_temp_mean_adj.contiguous().repeat(1, mean_temp_mean.shape[-1]).view(-1)
            Q_temp_var_adj = Q_temp_var_adj.contiguous().repeat(1, mean_temp_var.shape[-1]).view(-1)

            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)
            mean_temp_feat = mean_temp_feat.contiguous().view(-1)
            mean_temp_mean = mean_temp_mean.contiguous().view(-1)
            mean_temp_var = mean_temp_var.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            scale_temp_feat = torch.clamp(scale_temp_feat.contiguous().view(-1), min=1e-9)
            scale_temp_mean = torch.clamp(scale_temp_mean.contiguous().view(-1), min=1e-9)
            scale_temp_var = torch.clamp(scale_temp_var.contiguous().view(-1), min= 1e-9)
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
            Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
            Q_temp_mean = Q_temp_mean * (1.1 + torch.tanh(Q_temp_mean_adj))
            Q_temp_var = Q_temp_var * (1.1 + torch.tanh(Q_temp_var_adj))

            scaling = _scaling[N_start:N_end][indices].view(-1)  # [N_num*6]
            scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling, min_scaling, max_scaling = encoder_gaussian(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)
            min_scaling_list.append(min_scaling)
            max_scaling_list.append(max_scaling)
            scaling_list.append(scaling)

            temp_mask = _temp_mask[N_start:N_end][indices]  # {0, 1}  # [N_num, 1]
            temp_mask = temp_mask.repeat(1, self.num_temp_feat*self.temp_feat_dim).to(torch.bool).view(-1)  # [N_num* temp_feat_dim]
            temp_feat = _temp_feat[N_start:N_end][indices].view(-1, self.num_temp_feat*self.temp_feat_dim).view(-1)  # [N_num* temp_feat_dim]
            temp_feat = STE_multistep.apply(temp_feat, Q_temp_feat, self._temperal_anchor_feat.mean())
            temp_feat[~temp_mask] = 0.0  # set mask to 0
            torch.cuda.synchronize(); t0 = time.time()
            if temp_mask.sum() != 0:
                bit_temp_feat, min_temp_feat, max_temp_feat = encoder_gaussian(temp_feat[temp_mask], mean_temp_feat[temp_mask], scale_temp_feat[temp_mask], Q_temp_feat[temp_mask], file_name=temp_feat_b_name)
            else:
                bit_temp_feat, min_temp_feat, max_temp_feat = 0, 0, 0
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_temp_feat_list.append(bit_temp_feat)
            min_temp_feat_list.append(min_temp_feat)
            max_temp_feat_list.append(max_temp_feat)
            temp_feat_list.append(temp_feat)

            mask = _mask[N_start:N_end][indices]  # {0, 1}  # [N_num, K, 1]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
            offsets = _grid_offsets[N_start:N_end][indices].view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
            offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
            offsets[~mask] = 0.0
            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets, min_offsets, max_offsets = encoder_gaussian(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_offsets_list.append(bit_offsets)
            min_offsets_list.append(min_offsets)
            max_offsets_list.append(max_offsets)
            offsets_list.append(offsets)

            temp_mean_t = _mean_t[N_start:N_end][indices].view(-1)  # [N_num*2]
            temp_mean_t = STE_multistep.apply(temp_mean_t, Q_temp_mean, self._temp_mask[:, 1:3, 0].mean())  # align STE grid to training-time anchor (was whole-tensor mean incl. binary ch)
            torch.cuda.synchronize(); t0 = time.time()
            bit_temp_mean, min_mean_t, max_mean_t = encoder_gaussian(temp_mean_t, mean_temp_mean, scale_temp_mean, Q_temp_mean, file_name=mean_t_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_temp_mean_list.append(bit_temp_mean)
            min_mean_t_list.append(min_mean_t)
            max_mean_t_list.append(max_mean_t)
            mean_t_list.append(temp_mean_t)

            temp_var_t = _var_t[N_start:N_end][indices].view(-1)  # [N_num*2]
            temp_var_t = STE_multistep.apply(temp_var_t, Q_temp_var, self._temp_mask[:, 3:5, 0].mean())  # align STE grid to training-time anchor
            torch.cuda.synchronize(); t0 = time.time()
            bit_temp_var, min_var_t, max_var_t = encoder_gaussian(temp_var_t, mean_temp_var, scale_temp_var, Q_temp_var, file_name=var_t_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_temp_var_list.append(bit_temp_var)
            min_var_t_list.append(min_var_t)
            max_var_t_list.append(max_var_t)
            var_t_list.append(temp_var_t)

            torch.cuda.empty_cache()

        bit_anchor = N * 3 * anchor_round_digits
        bit_feat_list = [sum(bit_feat) for bit_feat in bit_feat_lists]
        total_bit_feat = sum(bit_feat_list)
        bit2MB_feat_list = [round(bit_feat/bit2MB_scale, 4) for bit_feat in bit_feat_list]
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)
        bit_temp_feat = sum(bit_temp_feat_list)
        bit_temp_mean = sum(bit_temp_mean_list)
        bit_temp_var = sum(bit_temp_var_list)

        indices = torch.cat(indices_list, dim=0)
        assert indices.shape[0] == _mask.shape[0]

        
        mask = _mask[indices]  # {0, 1}

        p = torch.zeros_like(mask).to(torch.float32)
        prob_masks = (mask.sum() / mask.numel()).item()
        p[...] = prob_masks
        bit_masks = encoder((mask * 2 - 1).view(-1), p.view(-1), file_name=masks_b_name)

        temp_masks = _temp_mask[indices]  # [N_num, 5]
        p = torch.zeros_like(temp_masks).to(torch.float32)
        prob_masks2 = (temp_masks.sum() / temp_masks.numel()).item()
        p[...] = prob_masks2
        bit_temp_masks = encoder((temp_masks * 2 - 1).view(-1), p.view(-1), file_name=temp_masks_b_name)

        show_feat_bits = ' '
        for i in range(self.num_feat_slices):
            show_feat_bits += f'feat{i} {round(bit_feat_list[i]/bit2MB_scale, 4)}, '

        tri_coder_list, triplane_mb=0,0
        triplane_mb=feat_rate / bit2MB_scale
        
        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"total_feat {round(total_bit_feat/bit2MB_scale, 4)}, " \
                   f"{show_feat_bits}" \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"temp_feat {round(bit_temp_feat/bit2MB_scale, 4)}, " \
                     f"temp_mean {round(bit_temp_mean/bit2MB_scale, 4)}, " \
                     f"temp_var {round(bit_temp_var/bit2MB_scale, 4)}, " \
                   f"temp_masks {round(bit_temp_masks/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Triplane_f {round(triplane_mb,4)}," \
                   f"Total {round((bit_anchor + total_bit_feat + bit_scaling + bit_offsets + bit_temp_feat + bit_masks + bit_temp_mean + bit_temp_var + self.get_mlp_size()[0])/bit2MB_scale + triplane_mb, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        rate_set=[round(bit_anchor/bit2MB_scale, 4), round(total_bit_feat/bit2MB_scale, 4)] + bit2MB_feat_list + [round(bit_scaling/bit2MB_scale, 4),round(bit_offsets/bit2MB_scale, 4), round(bit_temp_feat/bit2MB_scale, 4), round(bit_temp_mean/bit2MB_scale, 4), round(bit_temp_var/bit2MB_scale, 4), round(bit_temp_masks/bit2MB_scale, 4), round(bit_masks/bit2MB_scale, 4),round(self.get_mlp_size()[0]/bit2MB_scale, 4),triplane_mb]
        return [self._anchor.shape[0], N, MAX_batch_size, anchor_infos_list, min_feat_lists, max_feat_lists, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, min_temp_feat_list, max_temp_feat_list, min_mean_t_list, max_mean_t_list, min_var_t_list, max_var_t_list, prob_masks, prob_masks2], log_info , rate_set, tri_coder_list


    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, patched_infos, tri_coder_list, weighted_mask):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')
        [N_full, N, MAX_batch_size, anchor_infos_list, min_feat_lists, max_feat_lists, min_scaling_list, max_scaling_list, min_offsets_list, max_offsets_list, min_temp_feat_list, max_temp_feat_list,min_mean_t_list, max_mean_t_list, min_var_t_list, max_var_t_list, prob_masks, prob_masks2] = patched_infos
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []
        temp_feat_decoded_list = []
        mean_t_decoded_list = []
        var_t_decoded_list = []

        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        p = torch.zeros(size=[N, self.n_offsets, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks
        masks_decoded = decoder(p.view(-1), masks_b_name)  # {-1, 1}
        masks_decoded = (masks_decoded + 1) / 2  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        temp_masks_b_name = os.path.join(pre_path_name, 'temp_masks.b')

        p = torch.zeros(size=[N, 1], device='cuda').to(torch.float32)
        p[...] = prob_masks2
        temp_masks_decoded = decoder(p.view(-1), temp_masks_b_name)  # {-1, 1}
        temp_masks_decoded = (temp_masks_decoded + 1) / 2  # {0, 1}
        temp_masks_decoded = temp_masks_decoded.view(-1, 1)  # [N, 5, 1]
        
        mask_anchor = self.get_mask_anchor(weighted_mask)
        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []
        
        anchor_decoded = torch.load(os.path.join(pre_path_name, 'anchor.pkl')).cuda()
        for s in range(steps):
            min_feat_per_slice = [min_feat_lists[i][s] for i in range(self.num_feat_slices)]
            max_feat_per_slice = [max_feat_lists[i][s] for i in range(self.num_feat_slices)]
            
            min_scaling = min_scaling_list[s]
            max_scaling = max_scaling_list[s]
            min_offsets = min_offsets_list[s]
            max_offsets = max_offsets_list[s]
            min_temp_feat = min_temp_feat_list[s]
            max_temp_feat = max_temp_feat_list[s]

            N_num = min(MAX_batch_size, N - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)
            # sizes of MLPs is not included here
            feat_b_name_list = []
            for i in range(self.num_feat_slices):
                feat_b_name_list.append(os.path.join(pre_path_name, f'feat{i}.b').replace('.b', f'_{s}.b'))
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')
            temp_feat_b_name = os.path.join(pre_path_name, 'temp_feat.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2
            Q_temp_feat = 1
            Q_temp_mean = 0.1
            Q_temp_var = 0.1

            indices = torch.tensor(data=range(N_num), device='cuda', dtype=torch.long)  # [N_num]
            anchor_infos = None
            anchor_infos_list.append(anchor_infos)

            # encode feat
            feat_context, feat_rate, f1rate, f2rate, f3rate  = self.feature_net(anchor_decoded[N_start:N_end], itr=-2)
            means, scales, mean_scaling, scale_scaling, mean_offsets, scale_offsets, mean_temp_feat, scale_temp_feat, mean_temp_mean, scale_temp_mean, mean_temp_var, scale_temp_var, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, Q_temp_feat_adj, Q_temp_mean_adj, Q_temp_var_adj = \
                torch.split(feat_context, split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, self.temp_feat_dim*self.num_temp_feat, self.temp_feat_dim*self.num_temp_feat, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], dim=-1)
            means = list(torch.split(means, split_size_or_sections=self.chcm_slices_list, dim=-1))
            scales = list(torch.split(scales, split_size_or_sections=self.chcm_slices_list, dim=-1))
            Q_feat_list = [Q_feat * (1.1 + torch.tanh(Q_feat_adj.contiguous().repeat(1, ch).view(-1))) for ch in self.chcm_slices_list]
  
            for i in range(self.num_feat_slices):
                if i == 0:
                    mean = means[i].contiguous().view(-1)
                    scale = torch.clamp(scales[i].contiguous().view(-1), min=1e-9)
                    feat = decoder_gaussian(mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i], min_value=min_feat_per_slice[i], max_value=max_feat_per_slice[i])
                    feat = feat.view(N_num, self.chcm_slices_list[i])
                    decoded_feat = feat
                    
                    
                else:
                    dmean, dscale = torch.split(self.mlp_chcm_list[i-1](decoded_feat), split_size_or_sections=[self.chcm_slices_list[i], self.chcm_slices_list[i]], dim=-1)
                    mean = means[i] + dmean
                    scale = scales[i] + dscale
                    mean = mean.contiguous().view(-1)
                    scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
                    feat = decoder_gaussian(mean, scale, Q_feat_list[i], file_name=feat_b_name_list[i], min_value=min_feat_per_slice[i], max_value=max_feat_per_slice[i])
                    feat = feat.view(N_num, self.chcm_slices_list[i])
                    decoded_feat = torch.cat([decoded_feat, feat], dim=-1)
                    
            feat_decoded_list.append(decoded_feat)
            

            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_scaling = Q_scaling * (1.1 + torch.tanh(Q_scaling_adj))
            
            mean_scaling = mean_scaling.contiguous().view(-1)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scaling_decoded = decoder_gaussian(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, min_value=min_scaling, max_value=max_scaling)
            scaling_decoded = scaling_decoded.view(N_num, 6)  # [N_num, 6]
            scaling_decoded_list.append(scaling_decoded)

            Q_temp_feat_adj = Q_temp_feat_adj.contiguous().repeat(1, mean_temp_feat.shape[-1]).view(-1)
            Q_temp_feat = Q_temp_feat * (1.1 + torch.tanh(Q_temp_feat_adj))
            
            mean_temp_feat = mean_temp_feat.contiguous().view(-1)
            scale_temp_feat = torch.clamp(scale_temp_feat.contiguous().view(-1), min=1e-9)
            temp_mask_tmp = temp_masks_decoded[N_start:N_end].repeat(1, self.num_temp_feat*self.temp_feat_dim).view(-1).to(torch.bool)  # [N_num* num_temp_feat* temp_feat_dim]
            if temp_mask_tmp.sum() == 0:
                temp_feat_decoded_tmp = torch.zeros_like(mean_temp_feat)
                temp_feat_decoded_list.append(temp_feat_decoded_tmp.view(N_num, self.num_temp_feat*self.temp_feat_dim))
            else:
                temp_feat_decoded_tmp = decoder_gaussian(mean_temp_feat[temp_mask_tmp], scale_temp_feat[temp_mask_tmp], Q_temp_feat[temp_mask_tmp], file_name=temp_feat_b_name, min_value=min_temp_feat, max_value=max_temp_feat)
                temp_feat_decoded = torch.zeros_like(mean_temp_feat)
                temp_feat_decoded[temp_mask_tmp] = temp_feat_decoded_tmp
                temp_feat_decoded = temp_feat_decoded.view(N_num, -1).view(N_num, self.num_temp_feat*self.temp_feat_dim)  # [N_num, num_temp_feat, temp_feat_dim]
                temp_feat_decoded_list.append(temp_feat_decoded)

            
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
            Q_offsets = Q_offsets * (1.1 + torch.tanh(Q_offsets_adj))
            if self.chcm_for_offsets:
                dmean_offsets, dscale_offsets = torch.split(self.mlp_chcm_offsets(decoded_feat), split_size_or_sections=[3*self.n_offsets, 3*self.n_offsets], dim=-1)
                mean_offsets = mean_offsets + dmean_offsets
                scale_offsets = scale_offsets + dscale_offsets
            mean_offsets = mean_offsets.contiguous().view(-1)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, min_value=min_offsets, max_value=max_offsets)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)  # [N_num, K, 3]
            offsets_decoded_list.append(offsets_decoded)
            
            Q_temp_mean_adj = Q_temp_mean_adj.contiguous().repeat(1, mean_temp_mean.shape[-1]).view(-1)
            Q_temp_mean = Q_temp_mean * (1.1 + torch.tanh(Q_temp_mean_adj))
            mean_temp_mean = mean_temp_mean.contiguous().view(-1)
            scale_temp_mean = torch.clamp(scale_temp_mean.contiguous().view(-1), min=1e-9)
            temp_mean_t_decoded = decoder_gaussian(mean_temp_mean, scale_temp_mean, Q_temp_mean, file_name=os.path.join(pre_path_name, 'mean_t.b').replace('.b', f'_{s}.b'), min_value=min_mean_t_list[s], max_value=max_mean_t_list[s])
            temp_mean_t_decoded = temp_mean_t_decoded.view(N_num, 2)  # [N_num, 2]
            mean_t_decoded_list.append(temp_mean_t_decoded)

            Q_temp_var_adj = Q_temp_var_adj.contiguous().repeat(1, mean_temp_var.shape[-1]).view(-1)
            Q_temp_var = Q_temp_var * (1.1 + torch.tanh(Q_temp_var_adj))
            mean_temp_var = mean_temp_var.contiguous().view(-1)
            scale_temp_var = torch.clamp(scale_temp_var.contiguous().view(-1), min=1e-9)
            temp_var_t_decoded = decoder_gaussian(mean_temp_var, scale_temp_var, Q_temp_var, file_name=os.path.join(pre_path_name, 'var_t.b').replace('.b', f'_{s}.b'), min_value=min_var_t_list[s], max_value=max_var_t_list[s])
            temp_var_t_decoded = temp_var_t_decoded.view(N_num, 2)  # [N_num, 2]
            var_t_decoded_list.append(temp_var_t_decoded)
            


            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)
        temp_feat_decoded = torch.cat(temp_feat_decoded_list, dim=0)
        mean_t_decoded = torch.cat(mean_t_decoded_list, dim=0)
        var_t_decoded = torch.cat(var_t_decoded_list, dim=0)

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N_full, 3], device='cuda')
        _anchor_feat = torch.zeros(size=[N_full, self.feat_dim], device='cuda')
        _offset = torch.zeros(size=[N_full, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N_full, 6], device='cuda')
        _temp_feat = torch.zeros(size=[N_full, self.temp_feat_dim], device='cuda')
        _mask = torch.zeros(size=[N_full, self.n_offsets, 1], device='cuda')
        _temp_mask = torch.zeros(size=[N_full, 5], device='cuda')  # [N_full, 5, 1]
        combine_temp_masks = torch.cat([temp_masks_decoded, mean_t_decoded, var_t_decoded], dim=1)  # [N_full, 5, 1]
        
        _anchor[:N] = anchor_decoded
        _anchor_feat[:N] = feat_decoded
        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N] = masks_decoded
        _temp_mask[:N] = combine_temp_masks
        _temp_feat[:N] = temp_feat_decoded

        mask_anchor = self.get_mask_anchor(weighted_mask)
        temp_mask_anchor = self.get_temp_mask(weighted_mask)[:, 0, :].squeeze()
        
        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        assert self._anchor_feat.shape == _anchor_feat.shape
        
        self._anchor_feat = nn.Parameter(_anchor_feat)
        assert self._offset.shape == _offset.shape
        self._offset = nn.Parameter(_offset)
        # If change the following attributes, decoded_version must be set True
        self.decoded_version = True
        assert self.get_anchor.shape == _anchor.shape
        self._anchor = nn.Parameter(_anchor)
        assert self._scaling.shape == _scaling.shape
        self._scaling = nn.Parameter(_scaling)
        _temp_feat=_temp_feat.unsqueeze(1)
        assert self._temperal_anchor_feat.shape == _temp_feat.shape
        self._temperal_anchor_feat = nn.Parameter(_temp_feat)
        assert self._mask.shape == _mask.shape
        self._mask = nn.Parameter(_mask)
        assert self._temp_mask[:,:5,0].shape == _temp_mask.shape
        self._temp_mask[:, :5, 0] = nn.Parameter(_temp_mask)  
        
        


        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

