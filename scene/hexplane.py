import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from scene.synthesis import UniformNoiseQuantizer,STEQuantizer
from scene.arm import ArmMLP, get_flat_latent_and_context, compute_rate
from typing_extensions import TypedDict

class EncoderOutput(TypedDict):
    """Define the dictionary containing COOL-CHIC encoder output as a type."""
    rate_y: Tensor                     # Rate [1]
    mu: Optional[List[Tensor]]         # List of N [H_i, W_i] tensors, mu for each latent grid (can be None)
    scale: Optional[List[Tensor]]      # List of N [H_i, W_i] tensors, scale for each latent grid (can be None)
    latent: Optional[List[Tensor]]     # List of N [H_i, W_i] tensors, each latent grid (can be None)


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def con_normalize_aabb(pts, aabb, con_aabb):
    scale = (aabb[1] - aabb[0]) / (con_aabb[1] - con_aabb[0])
    normalized_pts = (pts - con_aabb[0]) * scale + aabb[0]
    return normalized_pts

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = -0.1,
        b: float = 0.1):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    assert grid_nd <= in_dim
    # Only select the combination for (0, 1) which corresponds to the xy-plane
    coo_combs = [(0, 1)]  # Explicitly select the xy-plane combination
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]  # Keep the resolution only for xy-plane
        ))
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    # Only use (0, 1) for xy-plane
    coo_combs = [(0, 1)]  # Explicitly select the xy-plane

    if num_levels is None:
        num_levels = len(ms_grids)

    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList

    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        coo_dict = []

        for ci, coo_comb in enumerate(coo_combs):
            # Interpolate only in the xy-plane (coo_comb is now always (0, 1))
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso

            # Perform grid sampling on the selected xy-plane (0, 1)
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )

            # Store the interpolated result for each plane
            coo_dict.append(interp_out_plane)
            interp_space = interp_space * interp_out_plane

        # Concatenate along feature dimension
        triplane_result = torch.cat(coo_dict, dim=1)
        concat_tri = True

        # Combine over scales
        if concat_features:
            if concat_tri:
                if scale_id != 0:
                    multi_scale_interp.append(triplane_result + multi_scale_interp[-1])
                else:
                    multi_scale_interp.append(triplane_result)
            else:
                if scale_id != 0:
                    multi_scale_interp.append(interp_space + multi_scale_interp[-1])
                else:
                    multi_scale_interp.append(interp_space)
        else:
            if concat_tri:
                multi_scale_interp = multi_scale_interp + triplane_result
            else:
                multi_scale_interp = multi_scale_interp + interp_space

    # Concatenate features if concat_features is True
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp
def positional_embedding(value, L=10):
    """
    Apply Fourier features (positional embedding) to the input scalar.
    
    Args:
    - value: Input scalar (e.g., z coordinate) as a PyTorch tensor.
    - L: Number of frequencies to use (controls the dimension of embedding).
    
    Returns:
    - Embedded high-dimensional vector for the input scalar (PyTorch tensor).
    """
    embed = []
    for i in range(L):
        embed.append(torch.sin(2**i * torch.pi * value).unsqueeze(-1))  # Add new dimension
        embed.append(torch.cos(2**i * torch.pi * value).unsqueeze(-1))  # Add new dimension
    return torch.cat(embed, dim=-1)  # Concatenate along the new dimension

class HexPlaneField(nn.Module):
    def __init__(
        self,
        
        bounds,
        planeconfig,
        multires,
        mode,
        contract,
        comp_iter,
        train_k,
        layers_arm: List = [16, 16, 16, 16],
        n_ctx_rowcol = 1,
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.total_bits = 0

        self.init_itr=comp_iter
        self.n_ctx_rowcol = n_ctx_rowcol
        print("n_ctx_rowcol:",self.n_ctx_rowcol)
        self.mask_size = 2 * n_ctx_rowcol + 1

        self.rotation = nn.Parameter(torch.eye(3), requires_grad=True)
        self.pca_mean = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.variance = nn.Parameter(torch.ones(3), requires_grad=True)
        self.con_aabb = nn.Parameter(torch.ones([2,3]),requires_grad=True)
        
        self.train_k = train_k
        config = self.grid_config[0].copy()
        self.resolutions = config["resolution"]
        # 1. Init planes
        self.feat_dim = 0
        self.feat_dim=20*3
        print("feature_dim:",self.feat_dim)
        self.quantizer = UniformNoiseQuantizer()
        self.ste_quantizer = STEQuantizer()
        self.log_2_encoder_gains = nn.Parameter(
            torch.arange(0., 3), requires_grad=True
        )
        non_zero_pixel_ctx = int(((2*n_ctx_rowcol+1) ** 2 - 1) / 2)
        self.non_zero_pixel_ctx_index = torch.arange(0, non_zero_pixel_ctx).cuda()
        
        self.contract = contract
    
        self.num_param = (config["resolution"][-1] ** 2) * 1.5 * config["output_coordinate_dim"][mode]

    def set_aabb(self,rotated_points,xyz_max, xyz_min):
        xyz_max = xyz_max.flatten().cpu().detach().numpy()
        xyz_min = xyz_min.flatten().cpu().detach().numpy()
        if self.contract == True:
            pts = self.contract_to_unisphere(rotated_points.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            xyz_min = torch.min(pts, dim=0, keepdim=True)[0].detach()
            xyz_max = torch.max(pts, dim=0, keepdim=True)[0].detach()
            self.con_aabb = nn.Parameter(torch.cat((xyz_min, xyz_max), dim=0),requires_grad=True)
            print("after contract:",xyz_min,xyz_max)
            xyz_max = [1.0, 1.0, 1.0]
            xyz_min = [-1.0, -1.0, -1.0]
        aabb = torch.tensor([
            xyz_min,
            xyz_max
        ], device='cuda')
        self.aabb = nn.Parameter(aabb,requires_grad=True)
        print("Voxel Plane: set aabb=",self.aabb)

    def set_rotation_matrix(self, rotation_matrix, mean, variance):
        self.rotation = nn.Parameter(rotation_matrix, requires_grad=True)
        self.pca_mean = nn.Parameter(mean, requires_grad=True)
        self.variance = nn.Parameter(variance, requires_grad=True)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None, itr=-1):
    
        if self.contract == True:
            pts = con_normalize_aabb(pts, self.aabb,self.con_aabb)
        else: 
            pts = normalize_aabb(pts, self.aabb)
        pex = positional_embedding(pts[:,0], L=10)
        pey = positional_embedding(pts[:,1], L=10)
        pe = positional_embedding(pts[:,2], L=10)
        combined = torch.cat([pex, pey, pe], dim=-1)
 
        return combined, 0,0,0,0

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None, itr=-1):
        if self.contract:
            pts = pts - self.pca_mean
            pts = torch.matmul(pts, self.rotation.detach())
            pts = pts / self.variance.detach()

            pts = self.contract_to_unisphere(pts.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
    
        features, a,b,c,d = self.get_density(pts, itr=itr)

        return features,a,b,c,d
    
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
        
    def grid_encode_forward(self,
            get_proba_param: bool = False,
            use_ste_quant: bool = True,
            AC_MAX_VAL: list = None
        )->EncoderOutput:

        self.non_zero_pixel_ctx_index = self.non_zero_pixel_ctx_index

        # ====================== Get sent latent codes ====================== #
        # Two different types of quantization. quantize() function uses the usual
        # noise addition proxy if self.training is True and the actual round
        # otherwise.
        # if use_ste_quant the straight-through estimator is used i.e. actual
        # quantization in the forward pass and gradient set to one in the backward.
        quantizer = self.ste_quantizer if use_ste_quant else self.uniform_noise_quantizer
        sent_latent = []
        for i, (cur_latent) in enumerate(self.grids):
            temp = []
            ii=0
            for latent in cur_latent:
                quant_step = 2 ** 4
                quant_latent = self.quantizer.apply(
                latent * quant_step, # Apply Q. step
                False                                           # Noise if needed
                )
                if quant_latent.size()[-1] == self.resolutions[-1] * 2:
                    q1 = quant_latent[..., :self.resolutions[-1], :self.resolutions[-1]]
                    q2 = quant_latent[..., self.resolutions[-1]:, :self.resolutions[-1]]
                    q3 = quant_latent[..., :self.resolutions[-1], self.resolutions[-1]:]
                    q4 = quant_latent[..., self.resolutions[-1]:, self.resolutions[-1]:]
                    quant_latent = torch.cat((q1, q2, q3, q4), dim=1)
                if ii==0:
                    if len(sent_latent) == 0:
                        sent_latent.append(quant_latent)
                    else:
                        sent_latent[0] = torch.cat((sent_latent[0], quant_latent), dim=1)
                
                ii = ii + 1
                
            
        # Clamp latent if we need to write a bitstream
        if AC_MAX_VAL != None:
            sent_latent = [
                torch.clamp(cur_latent, -AC_MAX_VAL[i], AC_MAX_VAL[i] + 1)
                for i, cur_latent in enumerate(sent_latent)
            ]
        
        # ====================== Get sent latent codes ====================== #

        # sent_latent
        flat_latents = []
        flat_contexts = []
        raw_proba_params = []
        rate_ys = []
        for i in range(len(sent_latent)):
            flat_latent, flat_context = get_flat_latent_and_context(
                (sent_latent[i]).unsqueeze(0), self.mask_size, self.non_zero_pixel_ctx_index
            )
            raw_proba_param = self.arm(flat_context)
            rate_y, _, _ = compute_rate(flat_latent, raw_proba_param, clamp=True)
            rate_ys.append(rate_y)
            flat_latents.append(flat_latent)
            flat_contexts.append(flat_context)
            raw_proba_params.append(raw_proba_param)
        
        rate_y1 = sum([rate_y.sum() for rate_y in rate_ys])

        if get_proba_param:
            mus = []
            scales = []
            latents = []
            for i in range(len(sent_latent)):
                _, flat_mu_y, flat_scale_y = compute_rate(flat_latents[i], raw_proba_params[i], clamp=True)
                mu = []
                scale = []
                latent = []

                # "Pointer" for the reading of the 1D scale, mu and rate
                cnt = 0
                h_i, w_i = sent_latent[i].size()[-2:]
                mu = flat_mu_y.view(-1, h_i, w_i)
                scale = flat_scale_y.view(-1, h_i, w_i)
                latent = sent_latent[i][0].view(-1, h_i, w_i)


                mus.append(mu)
                scales.append(scale)
                latents.append(latent)
        else:
            mu = None
            scale = None
            latent = None

        out: EncoderOutput = {
            'rate_y': rate_y1,
            'mu': mus,
            'scale': scales,
            'latent': latents,
            'total_bits': rate_y1
        }
        return out
