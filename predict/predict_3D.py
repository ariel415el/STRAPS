import os
import cv2
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import consts

from predict.predict_joints2D import predict_joints2D

from models.smpl_official import SMPL
from renderers.weak_perspective_pyrender_renderer import Renderer

from utils.image_utils import pad_to_square, crop_and_resize_silhouette_joints
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps
from utils.rigid_transform_utils import rot6d_to_rotmat

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from segment_anything import build_sam, SamPredictor


class SAM:
    """This is a bounding box conditioned open segmentor that segments the main object in the given bbox"""
    def __init__(self, device):
        sam_checkpoint = '/mnt/storage_ssd/big_files/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        self.device = device

    @staticmethod
    def draw_mask(mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        from PIL import Image
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def predict(self, image, boxe):
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        # boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxe.to(self.device), image.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        masks = masks[0][0].cpu().numpy().astype(np.uint8)
        return masks, SAM.draw_mask(masks, image)

def setup_detectron2_predictors(silhouettes_from, device):
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    kprcnn_cfg.MODEL['DEVICE'] = device
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

    if silhouettes_from=='detectron2':
        cfg = get_cfg()
        ckpt = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(ckpt))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ckpt)
        silhouette_predictor = DefaultPredictor(cfg)
    else:
        silhouette_predictor = SAM(device)


    return joints2D_predictor, silhouette_predictor


def create_proxy_representation(silhouette,
                                joints2D,
                                out_wh):

    heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     out_wh)
    proxy_rep = np.concatenate([silhouette[:, :, None], heatmaps], axis=-1)
    proxy_rep = np.transpose(proxy_rep, [2, 0, 1])  # (C, out_wh, out_WH)

    return proxy_rep


def predict_3D(opts, input,
               regressor,
               device,
               silhouettes_from='detectron2',
               proxy_rep_input_wh=512,
               save_proxy_vis=False,
               render_vis=False,
               outpath=None):

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from, device)

    # Set-up SMPL model.
    smpl = SMPL(consts.SMPL_MODEL_DIR, batch_size=1).to(device)

    if render_vis:
        # Set-up renderer for visualisation.
        wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    if os.path.isdir(input):
        image_fnames = [f for f in sorted(os.listdir(input)) if f.endswith('.png') or
                        f.endswith('.jpg')]
        for fname in image_fnames:
            gender = torch.zeros(1).to(device).long()
            print("Predicting on:", fname)
            image = cv2.imread(os.path.join(input, fname))
            # Pre-process for 2D detectors
            image = pad_to_square(image)
            image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)
            # Predict 2D
            joints2D, bbox, joints2D_vis = predict_joints2D(image, joints2D_predictor)
            if silhouettes_from == 'detectron2':
                from predict.predict_silhouette_detectron import predict_silhouette_detectron
                silhouette, silhouette_vis = predict_silhouette_detectron(image,
                                                                          silhouette_predictor)
            elif silhouettes_from == 'sam':
                silhouette, silhouette_vis = silhouette_predictor.predict(image, bbox)
            # Crop around silhouette
            silhouette, joints2D, image = crop_and_resize_silhouette_joints(silhouette,
                                                                            joints2D,
                                                                            out_wh=opts.regressor_input_dim,
                                                                            image=image,
                                                                            image_out_wh=proxy_rep_input_wh,
                                                                            bbox_scale_factor=1.2)
            # Create proxy representation
            proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                    out_wh=opts.regressor_input_dim)
            proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
            proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

            # Predict 3D
            regressor.eval()
            with torch.no_grad():
                pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep, gender) # TODO load gender
                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                pred_smpl_output = smpl(gender=gender, body_pose=pred_pose_rotmats[:, 1:],
                                        global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                        betas=pred_shape,
                                        pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
                pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                              proxy_rep_input_wh)

                pred_reposed_smpl_output = smpl(gender=gender, betas=pred_shape)
                pred_reposed_vertices = pred_reposed_smpl_output.vertices

            # Numpy-fying
            pred_vertices = pred_vertices.cpu().detach().numpy()[0]
            pred_vertices2d = pred_vertices2d.cpu().detach().numpy()[0]
            pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()[0]
            pred_cam_wp = pred_cam_wp.cpu().detach().numpy()[0]

            if outpath is None:
                outpath = input
            if not os.path.isdir(os.path.join(outpath, 'verts_vis')):
                os.makedirs(os.path.join(outpath, 'verts_vis'))
            plt.figure(figsize=(5,5))
            plt.imshow(silhouette_vis[..., :3])
            plt.scatter(pred_vertices2d[:, 0], pred_vertices2d[:, 1], s=0.3)
            plt.tight_layout(pad=0)
            plt.gca().set_axis_off()
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(os.path.join(outpath, 'verts_vis', 'verts_'+fname))

            if render_vis:
                # interactive_render(pred_vertices, pred_cam_wp)
                rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp, img=image)
                rend_reposed_img = wp_renderer.render(verts=pred_reposed_vertices,
                                                      cam=np.array([0.8, 0., -0.2]),
                                                      angle=180,
                                                      axis=[1, 0, 0])
                if not os.path.isdir(os.path.join(outpath, 'rend_vis')):
                    os.makedirs(os.path.join(outpath, 'rend_vis'))
                cv2.imwrite(os.path.join(outpath, 'rend_vis', 'rend_'+fname), rend_img)
                cv2.imwrite(os.path.join(outpath, 'rend_vis', 'reposed_'+fname), rend_reposed_img)

            if save_proxy_vis:
                if not os.path.isdir(os.path.join(outpath, 'proxy_vis')):
                    os.makedirs(os.path.join(outpath, 'proxy_vis'))
                cv2.imwrite(os.path.join(outpath, 'proxy_vis', 'silhouette_'+fname), silhouette_vis)
                cv2.imwrite(os.path.join(outpath, 'proxy_vis', 'joints2D_'+fname), joints2D_vis)

            def write_obj_file(vertices, faces, filename):
                with open(filename, 'w') as f:
                    # Write vertices
                    for vertex in vertices:
                        f.write(f"v {' '.join(map(str, vertex))}\n")

                    # Write faces
                    for face in faces:
                        f.write(f"f {' '.join(map(str, [index + 1 for index in face]))}\n")  #

            os.makedirs(os.path.join(outpath, 'objs'), exist_ok=True)
            write_obj_file(pred_vertices, np.load(consts.SMPL_FACES_PATH), os.path.join(outpath, 'objs', fname.split(".")[0]+".obj"))