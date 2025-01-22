import numpy as np
import torch

import consts
from smplx.lbs import batch_rodrigues

from augmentation.smpl_augmentation import augment_smpl
from augmentation.cam_augmentation import augment_cam_t
from renderers.nmr_renderer import NMRRenderer
from utils.cam_utils import perspective_project_torch, get_intrinsics_matrix
from augmentation.proxy_rep_augmentation import augment_proxy_representation, random_verts2D_deviation
from utils.image_utils import batch_crop_seg_to_bounding_box, batch_resize
from utils.label_conversions import convert_multiclass_to_binary_labels_torch, convert_2Djoints_to_gaussian_heatmaps_torch




class DataProcessor(object):
    def __init__(self, opts, smpl_model):
        self.opts = opts
        self.device = opts.device
        self.smpl_model = smpl_model
        self.set_renderer(opts)

        # Loading mean shape (for augmentation function)
        mean_smpl = np.load(consts.SMPL_MEAN_PARAMS_PATH)
        self.mean_shape = torch.from_numpy(mean_smpl['shape']).float().to(self.device)

    def set_renderer(self, opts):
        # Camera and NMR part/silhouette renderer
        # Assuming camera rotation is identity (since it is dealt with by global_orients in SMPL)
        device = opts.device
        batch_size = opts.batch_size
        mean_cam_t = np.array(opts.mean_camera_t)
        mean_cam_t = torch.from_numpy(mean_cam_t).float().to(device)
        self.mean_cam_t = mean_cam_t[None, :].expand(batch_size, -1)
        cam_K = get_intrinsics_matrix(opts.regressor_input_dim, opts.regressor_input_dim, opts.focal_length)
        cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
        cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
        cam_R = torch.eye(3).to(device)
        cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        self.cam_K = cam_K
        self.cam_R = cam_R
        self.nmr_parts_renderer = NMRRenderer(batch_size,
                                         self.cam_K,
                                         self.cam_R,
                                         opts.regressor_input_dim,
                                         rend_parts_seg=not self.opts.condition_on_depth).to(device)


    def prepare_pose(self, target_pose, target_shape, gender, augment=True):
        # TARGET SMPL PARAMETERS

        target_body_pose, target_orient = target_pose[:, 3:], target_pose[:, :3]
        if augment:
            # SMPL AND CAM AUGMENTATION
            target_shape, target_body_pose_rotmat, target_orient_rotmat = augment_smpl(
                target_shape,
                target_body_pose,
                target_orient,
                self.mean_shape,
                self.opts.smpl_augment_params)
            target_cam_t = augment_cam_t(self.mean_cam_t,
                                         xy_std=self.opts.cam_augment_params['xy_std'],
                                         delta_z_range=self.opts.cam_augment_params['delta_z_range'])
        else:
            target_cam_t = self.mean_cam_t
            target_body_pose_rotmat = batch_rodrigues(target_body_pose.contiguous().view(-1, 3))
            target_body_pose_rotmat = target_body_pose_rotmat.view(-1, 23, 3, 3)
            target_orient_rotmat = batch_rodrigues(target_orient.contiguous().view(-1, 3))
            target_orient_rotmat = target_orient_rotmat.unsqueeze(1)


        # TARGET VERTICES AND JOINTS
        target_smpl_output = self.smpl_model(gender=gender,
                                        body_pose=target_body_pose_rotmat,
                                        global_orient=target_orient_rotmat,
                                        betas=target_shape,
                                        pose2rot=False)

        target_pose_rotmat = torch.cat([target_orient_rotmat, target_body_pose_rotmat], dim=1)

        target_vertices = target_smpl_output.vertices
        target_joints_all = target_smpl_output.joints
        target_joints_h36m = target_joints_all[:, consts.ALL_JOINTS_TO_H36M_MAP, :]
        target_joints_h36mlsp = target_joints_h36m[:, consts.H36M_TO_J14, :]
        target_joints_coco = target_joints_all[:, consts.ALL_JOINTS_TO_COCO_MAP, :]
        target_joints2d_coco = perspective_project_torch(points=target_joints_coco,
                                                         rotation=self.cam_R,
                                                         translation=target_cam_t,
                                                         cam_K=self.cam_K)
        target_reposed_smpl_output = self.smpl_model(gender=gender, betas=target_shape)
        target_reposed_vertices = target_reposed_smpl_output.vertices

        if self.opts.proxy_rep_augment_params['deviate_verts2D']:
            # Vertex noise augmentation to give noisy proxy representation edges
            target_vertices_for_rendering = random_verts2D_deviation(target_vertices,
                                                                     delta_verts2d_dev_range=
                                                                     self.opts.proxy_rep_augment_params[
                                                                         'delta_verts2d_dev_range'])
        else:
            target_vertices_for_rendering = target_vertices

        # INPUT PROXY REPRESENTATION GENERATION
        input = self.nmr_parts_renderer(target_vertices_for_rendering, target_cam_t)

        # BBOX AUGMENTATION AND CROPPING
        if self.opts.bbox_augment_params['crop_input']:
            # Crop inputs according to bounding box
            # + add random scale and centre augmentation
            input = input.cpu().detach().numpy()
            target_joints2d_coco = target_joints2d_coco.cpu().detach().numpy()
            all_cropped_segs, all_cropped_joints2D = batch_crop_seg_to_bounding_box(
                input, target_joints2d_coco,
                orig_scale_factor=self.opts.bbox_augment_params['mean_scale_factor'],
                delta_scale_range=self.opts.bbox_augment_params['delta_scale_range'] if augment else None,
                delta_centre_range=self.opts.bbox_augment_params['delta_centre_range'] if augment else None)
            resized_input, resized_joints2D = batch_resize(all_cropped_segs,
                                                           all_cropped_joints2D,
                                                           self.opts.regressor_input_dim)
            input = torch.from_numpy(resized_input).float().to(self.device)
            target_joints2d_coco = torch.from_numpy(resized_joints2D).float().to(self.device)

        if augment:
            # PROXY REPRESENTATION AUGMENTATION
            input, target_joints2d_coco = augment_proxy_representation(input,
                                                                             target_joints2d_coco,
                                                                             self.opts.proxy_rep_augment_params)

        # FINAL INPUT PROXY REPRESENTATION GENERATION WITH JOINT HEATMAPS
        if not self.opts.condition_on_depth:
            input = convert_multiclass_to_binary_labels_torch(input)

        input = input.unsqueeze(1)
        j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco,
                                                                   self.opts.regressor_input_dim)
        input = torch.cat([input, j2d_heatmaps], dim=1)

        return input, target_pose_rotmat, target_joints2d_coco, target_vertices, target_joints_h36mlsp, target_reposed_vertices