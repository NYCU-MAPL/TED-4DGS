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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.loader = ""  # "" = n3dv/baseline path; "nerfies" = hypernerf(DSGC) path — set by --config arguments/<scene>.py
        self.shuffle = False  # hyper(DSGC) Scene shuffle flag; n3dv path ignores it
        self.feat_dim = 50
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._ckpt_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8
        self.lod = 0
        self.temp_embedding = 0
        self.temporal_embedding_dim=256
        self.anchor_embedding_dim=8
        self.k_frames_for_one_feature=300
        self.attribute_config = {
            'net_width': 100,
            'net_depth': 0,
            'final_dim': (6+30+self.feat_dim+self.anchor_embedding_dim + 2+ 2)*2+1+1+1+1+1+1,
            'bounds': 1,
            'kplanes_config': {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 3,
                             'output_coordinate_dim':  {'sof': 8},
                             'resolution': [1, 1, 1]
                            },
            'if_contract': False,
            'comp_iter':0,
            'multires': [2, 4],
            
        }

        # Per-Gaussian E-D3DGS
        self.net_width = 128
        self.defor_depth = 1
        self.min_embeddings = 30
        self.max_embeddings = 150
        self.no_ds=False
        self.no_dr=False
        self.no_do=True
        self.no_dc=True
        
        
        self.use_coarse_temporal_embedding=False
        self.no_c2f_temporal_embedding=False
        self.no_coarse_deform=False
        self.no_fine_deform=False
        
        self.total_num_frames=300
        self.c2f_temporal_iter=10000
        self.deform_from_iter=5000
        self.use_anneal=True
        self.zero_temporal=False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.batch_size = 1  # n3dv baseline default (was CLI arg); hyper(DSGC) configs set 2
        self.compress_max_iterations = 80_001  # hyper(DSGC) only
        self.deformation_lr_max_steps = 60_000  # hyper(DSGC) only
        self.dataloader = True

        self.iterations = 80_000
        self.lr_max_iterations = 80_000
        self.comp_lr_max_iterations = 100_000
        self.scaffold_mlp_lr_max_iterations = 60_000

        self.triplane_init_fit_iter = 80001

        # for anchor densification
        self.start_stat = 500
        self.update_from = 5000
        self.update_interval = 300
        self.update_until = 60000

        
        self.position_lr_init = 0
        self.position_lr_final = 0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.scaffold_mlp_lr_max_iterations
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = self.scaffold_mlp_lr_max_iterations
        
        self.mask_lr_init = 0.007
        self.mask_lr_final = 0.001
        self.mask_lr_delay_mult = 0.01
        self.mask_lr_start_steps = 60000
        self.mask_lr_max_steps = 80000

        self.feature_lr = 0.0075

        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = self.scaffold_mlp_lr_max_iterations

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = self.scaffold_mlp_lr_max_iterations
        
        self.mlp_color_lr_init = 0.002
        self.mlp_color_lr_final = 0.00002
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = self.scaffold_mlp_lr_max_iterations

        self.inr_hyper_lr_start_steps = self.triplane_init_fit_iter
        self.inr_hyper_lr_max_steps = self.comp_lr_max_iterations
        self.arm_lr_start_steps = self.triplane_init_fit_iter + 5000
        self.arm_lr_max_steps = self.comp_lr_max_iterations

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = self.lr_max_iterations

        self.encoding_xyz_lr_init = 0.005
        self.encoding_xyz_lr_final = 0.00001
        self.encoding_xyz_lr_delay_mult = 0.33
        self.encoding_xyz_lr_max_steps = self.lr_max_iterations

        self.mlp_grid_lr_init = 0.0016
        self.mlp_grid_lr_final = 0.00016
        self.mlp_grid_lr_delay_mult = 0.01
        self.mlp_grid_lr_max_steps = self.lr_max_iterations

        self.mlp_deform_lr_init = 0.00016
        self.mlp_deform_lr_final = 0.000016
        self.mlp_deform_lr_delay_mult = 0.01
        self.mlp_deform_lr_max_steps = self.lr_max_iterations

        self.temp_feat_lr_init = 0.0025
        self.temp_feat_lr_final = 0.0025
        self.temp_feat_lr_delay_mult = 0.01
        self.temp_feat_lr_max_steps = self.lr_max_iterations

        self.temp_mask_lr_init = 0.01
        self.temp_mask_lr_final = 0.0001
        self.temp_mask_lr_delay_mult = 0.01
        self.temp_mask_lr_max_steps = self.lr_max_iterations
        self.temp_mask_lr_start_steps = 0

        self.ic_lr_init = 1e-4
        self.ic_lr_final = 1e-5
        self.ic_lr_delay_mult = 0.01
        self.ic_lr_max_steps = 80000
        self.ic_lr_start_steps = 5000

        self.entropy_model_lr_init = 0.001
        self.entropy_model_lr_final = 0.0001
        self.entropy_model_lr_delay_mult = 0.01
        self.entropy_model_lr_max_steps = 120000
        self.entropy_model_lr_step_sub = 100000

        self.hyper_lr_init = 0.0075
        self.hyper_lr_final = 0.0075
        self.hyper_lr_delay_mult = 0.01
        self.hyper_lr_max_steps = 120000
        self.hyper_lr_start_steps = 100000

        self.percent_dense = 0.01
        self.lambda_dssim = 1/3
        self.opacity_l1_coef_fine = 0

        self.use_tv_loss = False
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.002 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid

        
        self.min_opacity = 0.005  # 0.2
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        self.time_densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
