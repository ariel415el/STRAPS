import os
import argparse
import torch

from models.regressor import SingleInputRegressor
from opts import Opts
import os
import cv2
import numpy as np
import matplotlib

from proxy_predictors.predict_silhouette import SAM

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


from smplx.lbs import batch_rodrigues
import consts
from proxy_predictors.predict_joints2D import predict_joints2D
from models.smpl_official import SMPL
from utils.image_utils import pad_to_square, crop_and_resize_silhouette_joints
from utils.cam_utils import orthographic_project_torch
from utils.joints2d_utils import undo_keypoint_normalisation
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps
from utils.rigid_transform_utils import rot6d_to_rotmat



class Predictor:
    def __init__(self, ckpt_path, regressor_input_dim, device, proxy_rep_input_wh=512):
        self.regressor = SingleInputRegressor(resnet_in_channels=18,
                                         resnet_layers=18,
                                         ief_iters=3)
        self.device = device
        self.regressor_input_dim = regressor_input_dim
        self.proxy_rep_input_wh = proxy_rep_input_wh
        print("Regressor loaded. Weights from:", ckpt_path)
        self.regressor.to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.regressor.load_state_dict(checkpoint['model_state_dict'])

        # Set-up SMPL model.
        self.smpl = SMPL(consts.SMPL_MODEL_DIR, batch_size=1).to(device)

        self.joints2D_predictor, self.silhouette_predictor = setup_detectron2_predictors(device)

    def infer_proxy_from_image(self, fpath):
        gender = torch.zeros(1).to(device).long() # TODO make dynamic
        image = cv2.imread(fpath)
        image = pad_to_square(image)
        image = cv2.resize(image, (self.proxy_rep_input_wh, self.proxy_rep_input_wh),
                           interpolation=cv2.INTER_LINEAR)

        joints2D, bbox, joints2D_vis = predict_joints2D(image, self.joints2D_predictor)
        silhouette, silhouette_vis = self.silhouette_predictor.predict(image, bbox)

        silhouette, joints2D, image = crop_and_resize_silhouette_joints(silhouette,
                                                                        joints2D,
                                                                        out_wh=self.regressor_input_dim,
                                                                        image=image,
                                                                        image_out_wh=self.proxy_rep_input_wh,
                                                                        bbox_scale_factor=1.2)
        # Create proxy representation
        proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                out_wh=self.regressor_input_dim)
        proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
        proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

        # Predict 3D
        self.regressor.eval()
        with torch.no_grad():
            pred_cam_wp, pred_pose, pred_shape = self.regressor(proxy_rep, gender)  # TODO load gender
            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            pred_smpl_output = self.smpl(gender=gender, body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_vertices2d = orthographic_project_torch(pred_vertices, pred_cam_wp)
            pred_vertices2d = undo_keypoint_normalisation(pred_vertices2d,
                                                          self.proxy_rep_input_wh)

        return silhouette_vis, pred_vertices, pred_vertices2d, joints2D_vis


def setup_detectron2_predictors(device):
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    kprcnn_cfg.MODEL['DEVICE'] = device
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

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

def save_to_file(silhouette_vis, pred_vertices, pred_vertices2d, joints2D_vis, outpath):
    os.makedirs(outpath, exist_ok=True)
    # Numpy-fying
    pred_vertices = pred_vertices.cpu().detach().numpy()[0]
    pred_vertices2d = pred_vertices2d.cpu().detach().numpy()[0]

    if not os.path.isdir(os.path.join(outpath, 'verts_vis')):
        os.makedirs(os.path.join(outpath, 'verts_vis'))
    plt.figure(figsize=(5, 5))
    plt.imshow(silhouette_vis[..., :3])
    plt.scatter(pred_vertices2d[:, 0], pred_vertices2d[:, 1], s=0.3)
    plt.tight_layout(pad=0)
    plt.gca().set_axis_off()
    plt.savefig(os.path.join(outpath, 'verts_vis', 'verts_' + fname))

    # Save proxies visualization
    if not os.path.isdir(os.path.join(outpath, 'proxy_vis')):
        os.makedirs(os.path.join(outpath, 'proxy_vis'))
    cv2.imwrite(os.path.join(outpath, 'proxy_vis', 'silhouette_' + fname), silhouette_vis)
    cv2.imwrite(os.path.join(outpath, 'proxy_vis', 'joints2D_' + fname), joints2D_vis)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    regressor_input_dim = Opts().regressor_input_dim # TODO read from ckpt folder
    predictor = Predictor(args.checkpoint, regressor_input_dim, device)

    out_path = os.path.join(os.path.dirname(args.checkpoint), "inference")
    for fname in os.listdir(args.input):
        print("Predicting on:", fname)
        fpath = os.path.join(args.input, fname)
        silhouette_vis, pred_vertices, pred_vertices2d, joints2D_vis = predictor.infer_proxy_from_image(fpath)
        save_to_file(silhouette_vis, pred_vertices, pred_vertices2d, joints2D_vis, out_path)

