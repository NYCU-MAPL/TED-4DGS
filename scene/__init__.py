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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from PIL import Image
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosHyper
from utils.graphics_utils import recordpointshelper
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=None,
                 resolution_scales=[1.0], load_coarse=False, opt=None, duration=None, loader=None, testonly=None):
        """b
        :param path: Path to colmap scene main folder.

        loader=="" (default, n3dv/baseline): auto-detect dataset type, FourDGSdataset wrappers.
        loader=="nerfies"/"dynerf"/"technicolor" (hyper/DSGC, set via --config): explicit loader branches.
        """
        if loader is None:
            loader = getattr(args, 'loader', "")
        self.loader = loader
        self.model_path = args.model_path

        if loader == "":
            # ================= n3dv / baseline path (unchanged) =================
            self.load_ckpt_path = args.ckpt_path
            self.gaussians = gaussians


            if load_iteration:
                if load_iteration == -1:
                    self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                else:
                    self.loaded_iter = load_iteration
                print("Loading trained model at iteration {}".format(self.loaded_iter))
            else:
                self.loaded_iter = None

            self.train_cameras = {}
            self.test_cameras = {}
            self.video_cameras = {}
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
                dataset_type="colmap"
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
                dataset_type="blender"
            elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
                scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
                dataset_type="dynerf"
            elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
                scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
                dataset_type="nerfies"
            elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
                scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
                dataset_type="PanopticSports"
            elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
                scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
                dataset_type="MultipleView"
            else:
                assert False, "Could not recognize scene type!"
            self.maxtime = scene_info.maxtime
            self.dataset_type = dataset_type
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            print("Loading Training Cameras")
            self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
            print("Loading Test Cameras")
            self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
            print("Loading Video Cameras")
            self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)
            print("Loading Done")

            xyz_max = scene_info.point_cloud.points.max(axis=0)
            xyz_min = scene_info.point_cloud.points.min(axis=0)
            if args.add_points:
                print("add points.")
                scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
            if self.loaded_iter:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, 300)
                self.gaussians.update_anchor_bound()
                self.gaussians.load_ply_sparse_gaussian(os.path.join(self.load_ckpt_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))
                if self.loaded_iter >= opt.triplane_init_fit_iter:
                    gaussians.set_feature_net()
                    gaussians.training_setup_triplane(opt)
                self.gaussians.load_mlp_checkpoints(os.path.join(self.load_ckpt_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "mlp_checkpoints.pth"
                                                       ))
            else:
                num_frames = self.get_num_frames()
                print("Creating Gaussian Model with {} frames".format(num_frames))
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, num_frames)
                self.gaussians.update_anchor_bound()
            return

        # ================= hyper (DSGC) path (unchanged from DSGC) =================
        if duration is None:
            duration = args.total_num_frames
        if shuffle is None:
            shuffle = getattr(args, 'shuffle', False)
        self.ckpt_path = args.ckpt_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.refmodelpath = None
        self.maxtime = duration

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        resolution_scales= [1.0]
        raydict = {}

        if loader == "dynerf":
            scene_info = sceneLoadTypeCallbacks["Dynerf"](args.source_path, args.source_path, args.eval, duration=300)
            print(f"dataset type: dynerf")
        elif loader == "technicolor" or loader == "technicolorvalid" :
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, duration=50.0, testonly=testonly)
        elif loader == "nerfies":
            scene_info = sceneLoadTypeCallbacks["Nerfies"](args.source_path, False, args.eval)
            print(f"dataset type: nerfies")
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            if loader in ["technicolorvalid"]:         
                self.train_cameras[resolution_scale] = [] # no training data
            elif loader in ["nerfies"]:
                self.train_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.train_cameras, resolution_scale, args)
            else: 
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
            
            
            print("Loading Test Cameras")
            if loader  in ["technicolorvalid", "technicolor", "dynerf", "dynerfvalid"]: # we need gt for metrics
                self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)
            elif loader in ["nerfies"]:
                self.test_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.test_cameras, resolution_scale, args)

            print("Loading Video Cameras")
            if loader  in ["technicolorvalid", "technicolor", "dynerf", "dynerfvalid"]: # we need gt for metrics
                self.video_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.video_cameras, resolution_scale, args)
            elif loader in ["nerfies"]:
                self.video_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.video_cameras, resolution_scale, args)


        if loader not in ["nerfies", "dynerf"]:
            for cam in self.test_cameras[resolution_scale]:
                if cam.image_name[:4] not in raydict and cam.rayo is not None:
                    raydict[cam.image_name[:4]] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

            for cam in self.test_cameras[resolution_scale]:
                cam.rays = raydict[cam.image_name[:4]] # should be direct ?

            if not testonly:
                for cam in self.train_cameras[resolution_scale]:
                    if cam.image_name[:4] not in raydict and cam.rayo is not None:
                        raydict[cam.image_name[:4]] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

                for cam in self.train_cameras[resolution_scale]:
                    cam.rays = raydict[cam.image_name[:4]] # should be direct ?


        if self.loaded_iter :
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, 50)
            self.gaussians.update_anchor_bound()
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.ckpt_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.ckpt_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "mlp_checkpoints.pth"
                                                   ))
        else:
            print("Creating Gaussian Model with {} frames".format(50))
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, 50)
            self.gaussians.update_anchor_bound()


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(os.path.join(point_cloud_path, "mlp_checkpoints.pth"))

    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def getTrainCameras(self, scale=1.0, data_loader=False):
        if self.loader:
            return self.train_cameras[scale]
        if data_loader:
            return self.train_camera
        else:
            return [self.train_camera[idx] for idx in range(len(self.train_camera))]

    def getValTrainCameras(self, scale=1.0):
        if self.loader:
            return [cam for ii, cam in enumerate(self.train_cameras[scale]) if ii % 23 == 0]
        return [self.train_camera[idx % len(self.train_camera)] for idx in range(10, 5999, 300)]

    def getOneTestCamera(self, scale=1.0):
        return [self.test_camera[idx] for idx in range(5, 300, 30)]

    def getTestCameras(self, scale=1.0):
        if self.loader:
            return self.test_cameras[scale]
        return [self.test_camera[idx] for idx in range(len(self.test_camera))]

    def getVideoCameras(self, scale=1.0):
        if self.loader:
            return self.video_cameras[scale]
        return self.video_camera

    def get_num_frames(self):
        num_frames = 300
        print("Number of frames: ", num_frames)
        return int(num_frames)
