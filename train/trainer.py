import os

import torch
import torch.optim as optim
from smplx.lbs import batch_rodrigues
from torch.utils.data import DataLoader
from tqdm import tqdm

import consts
from data.data_process import DataProcessor

from data.synthetic_training_dataset import SyntheticTrainingDataset

from models.regressor import SingleInputRegressor
from models.smpl_official import SMPL
from opts import Opts

from losses.multi_task_loss import HomoscedasticUncertaintyWeightedMultiTaskLoss
from utils.joints2d_utils import check_joints2d_visibility_torch

from utils.cam_utils import orthographic_project_torch

from utils.rigid_transform_utils import rot6d_to_rotmat
from wandb_logger import WandbLogger


class Trainer(object):
    def __init__(self):
        self.opts = Opts()
        self._set_dirs(self.opts.experiment_name)
        self._set_datasets(self.opts.val_perc)

        self.regressor = SingleInputRegressor(self.opts.resnet_in_channels, self.opts.resnet_layers,
                                              ief_iters=self.opts.ief_iters).to(self.opts.device)
        self.smpl_model = SMPL(consts.SMPL_MODEL_DIR, batch_size=self.opts.batch_size).to(self.opts.device)

        self.criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(self.opts.losses_on, self.opts.regressor_input_dim,
                                                                  init_loss_weights=self.opts.init_loss_weights,
                                                                  reduction='mean').to(self.opts.device)

        self.optimiser = optim.Adam(list(self.regressor.parameters()) + list(self.criterion.parameters()), lr=self.opts.lr)

        self.proxy_creator = DataProcessor(self.opts, self.smpl_model)

        self.logger = WandbLogger(self.opts.save_val_metrics, project_name="STAPS", run_name=self.opts.experiment_name)

        self.global_step = 0
        self.epoch = 0

    def _set_datasets(self, val_perc):
        train_dataset = SyntheticTrainingDataset(npz_path=consts.TRAIN_DATA_PATH, params_from='all')

        # Split train val
        n = len(train_dataset)
        val_size = int(val_perc * n)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n - val_size, val_size])

        def get_dataloader(dataset, opts):
            return DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                              drop_last=True, num_workers=opts.num_workers,
                              pin_memory=opts.pin_memory)

        self.train_dataloader = get_dataloader(train_dataset, self.opts)
        self.val_dataloader = get_dataloader(val_dataset, self.opts)

    def _set_dirs(self, experiment_name):
        # Path to save model weights to (without .tar extension).
        self.model_save_path = os.path.join(f'trained_models/{experiment_name}/straps_model_checkpoint_exp001')
        if not os.path.isdir(f'trained_models/{experiment_name}'):
            os.makedirs(f'trained_models/{experiment_name}')
        if not os.path.isdir('./logs'):
            os.makedirs('./logs')

    def process_batch(self, batch, augment):
        target_pose = batch['pose'].to(self.opts.device)
        target_shape = batch['shape'].to(self.opts.device)
        gender = torch.randint(0, 2, (len(target_pose),)).to(self.opts.device)

        with torch.no_grad():
            input, target_pose_rotmat, target_joints2d_coco, \
                target_vertices, target_joints_h36mlsp, target_reposed_vertices \
                = self.proxy_creator.prepare_pose(target_pose, target_shape, gender, augment=augment)

        # ---------------- FORWARD PASS ----------------
        # (gradients being computed from here on)
        pred_cam_wp, pred_pose, pred_shape = self.regressor(input, gender)

        # ---------------- process predictions ----------------
        pred_vertices, pred_pose_rotmats, pred_joints2d_coco, \
            pred_joints_h36mlsp, pred_reposed_vertices \
            = process_prediction(pred_pose, pred_shape, pred_cam_wp, gender, self.proxy_creator)

        # ---------------- LOSS ----------------
        # Check joints visibility
        target_joints2d_vis_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                   self.opts.regressor_input_dim)

        pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                              'verts': pred_vertices,
                              'shape_params': pred_shape,
                              'pose_params_rot_matrices': pred_pose_rotmats,
                              'joints3D': pred_joints_h36mlsp}
        target_dict_for_loss = {'joints2D': target_joints2d_coco,
                                'verts': target_vertices,
                                'shape_params': target_shape,
                                'pose_params_rot_matrices': target_pose_rotmat,
                                'joints3D': target_joints_h36mlsp,
                                'vis': target_joints2d_vis_coco}

        return self.criterion(target_dict_for_loss, pred_dict_for_loss)

    def train(self):
        for self.epoch in range(self.opts.num_epochs):
            print('\nEpoch {}/{}'.format(self.epoch, self.opts.num_epochs - 1))
            print('-' * 10)

            # Train
            self.regressor.train()
            for batch in tqdm(self.train_dataloader):
                loss, losses_dict = self.process_batch(batch, augment=True)
                loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()

                losses_dict['loss'] = loss
                self.logger.log(losses_dict, step=self.global_step, train=True)
                self.global_step += 1

            # Eval
            self.regressor.eval()
            with torch.no_grad():
                aggregated_losses = None
                for batch in tqdm(self.train_dataloader):
                    loss, losses_dict = self.process_batch(batch, augment=True)

                    losses_dict['loss'] = loss
                    if aggregated_losses is None:
                        aggregated_losses = losses_dict
                    else:
                        for key in losses_dict:
                            aggregated_losses[key] += (losses_dict[key] / len(self.train_dataloader))

            self.logger.log(aggregated_losses, step=self.global_step, train=False)

            #### # Save model
            save_dict = {'epoch': self.epoch,
                         'global_step': self.global_step,
                         'model_state_dict': self.regressor.state_dict(),
                         'optimiser_state_dict': self.optimiser.state_dict(),
                         'criterion_state_dict': self.criterion.state_dict()}
            torch.save(save_dict, self.model_save_path + f'_epoch-{self.epoch}.pt')

            # if new_best:
            #     print("Best epoch val metrics updated to ", best_epoch_val_metrics)
            #     torch.save(save_dict, self.model_save_path + f'_best.tar')

            # TODO: this causes some segfault
            # predict_3D(os.path.join(os.path.dirname(__file__),'..', 'demo'),
            #            regressor.cpu(), torch.device("cpu"),
            #            silhouettes_from='sam',
            #            outpath=model_save_path + f'test_epoch{epoch}')
            # self.regressor.to(device)

        # print('Training Completed. Best Val Metrics:\n')
        # return best_model_wts


def process_prediction(pred_pose, pred_shape, pred_cam_wp, gender, proxy_creator):
    # Convert pred pose to rotation matrices
    if pred_pose.shape[-1] == 24 * 3:
        pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
        pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
    elif pred_pose.shape[-1] == 24 * 6:
        pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

    # PREDICTED VERTICES AND JOINTS
    pred_smpl_output = proxy_creator.smpl_model(gender=gender,
                                                body_pose=pred_pose_rotmats[:, 1:],
                                                global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                                betas=pred_shape,
                                                pose2rot=False)
    pred_vertices = pred_smpl_output.vertices
    pred_joints_all = pred_smpl_output.joints
    pred_joints_h36m = pred_joints_all[:, consts.ALL_JOINTS_TO_H36M_MAP, :]
    pred_joints_h36mlsp = pred_joints_h36m[:, consts.H36M_TO_J14, :]
    pred_joints_coco = pred_joints_all[:, consts.ALL_JOINTS_TO_COCO_MAP, :]
    pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_cam_wp)
    pred_reposed_smpl_output = proxy_creator.smpl_model(gender=gender, betas=pred_shape)
    pred_reposed_vertices = pred_reposed_smpl_output.vertices

    return pred_vertices, pred_pose_rotmats, pred_joints2d_coco, pred_joints_h36mlsp, pred_reposed_vertices
