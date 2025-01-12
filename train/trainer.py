import copy
import os

import torch
import numpy as np
import torch.optim as optim
from smplx.lbs import batch_rodrigues
from torch.utils.data import DataLoader
from tqdm import tqdm

import consts
from data.data_process import DataProcessor

from data.synthetic_training_dataset import SyntheticTrainingDataset
from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

from models.regressor import SingleInputRegressor
from models.smpl_official import SMPL
from opts import Opts

from losses.multi_task_loss import HomoscedasticUncertaintyWeightedMultiTaskLoss
from utils.joints2d_utils import check_joints2d_visibility_torch

from utils.cam_utils import orthographic_project_torch

from utils.rigid_transform_utils import rot6d_to_rotmat


class Trainer(object):
    def __init__(self):
        self.opts = Opts()
        self.set_dirs(self.opts.experiment_name)
        self.set_datasets(self.opts.val_perc)

        self.regressor = SingleInputRegressor(self.opts.resnet_in_channels, self.opts.resnet_layers,
                                              ief_iters=self.opts.ief_iters).to(self.opts.device)
        self.smpl_model = SMPL(consts.SMPL_MODEL_DIR, batch_size=self.opts.batch_size).to(self.opts.device)

        self.criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(self.opts.losses_on, self.opts.regressor_input_dim,
                                                                  init_loss_weights=self.opts.init_loss_weights,
                                                                  reduction='mean').to(self.opts.device)
        self.optimiser = optim.Adam(list(self.regressor.parameters()) + list(self.criterion.parameters()), lr=self.opts.lr)

        self.proxy_creator = DataProcessor(self.opts, self.smpl_model)


    def set_datasets(self, val_perc):
        train_dataset = SyntheticTrainingDataset(npz_path=consts.TRAIN_DATA_PATH, params_from='all')
        n = len(train_dataset)
        val_size = int(val_perc * n)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n - val_size, val_size])

        def get_dataloader(dataset, opts):
            return DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                              drop_last=True, num_workers=opts.num_workers,
                              pin_memory=opts.pin_memory)

        self.train_dataloader = get_dataloader(train_dataset, self.opts)
        self.val_dataloader = get_dataloader(val_dataset, self.opts)

    def set_dirs(self, experiment_name):
        # Path to save model weights to (without .tar extension).
        self.model_save_path = os.path.join(f'trained_models/{experiment_name}/straps_model_checkpoint_exp001')
        self.log_path = os.path.join(f'trained_models/{experiment_name}/straps_model_logs_exp001.pkl')
        if not os.path.isdir(f'trained_models/{experiment_name}'):
            os.makedirs(f'trained_models/{experiment_name}')
        if not os.path.isdir('./logs'):
            os.makedirs('./logs')

    def run_epoch(self, dataloader, metrics_tracker, train=True):
        for batch_num, samples_batch in enumerate(tqdm(dataloader)):
            target_pose = samples_batch['pose'].to(self.opts.device)
            target_shape = samples_batch['shape'].to(self.opts.device)
            gender = torch.randint(0, 2, (len(target_pose),)).to(self.opts.device)
            num_train_inputs_in_batch = target_pose.shape[0]  # Same as bs since drop_last=True

            input, target_pose_rotmat, target_joints2d_coco, \
                target_vertices, target_joints_h36mlsp, target_reposed_vertices \
                = self.proxy_creator.prepare_pose(target_pose, target_shape, gender, augment=train)

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


            if train:
                self.optimiser.zero_grad()

            loss, task_losses_dict = self.criterion(target_dict_for_loss, pred_dict_for_loss)
            # ---------------- BACKWARD PASS ----------------
            if train:
                loss.backward()
                self.optimiser.step()

            # ---------------- TRACK LOSS AND METRICS ----------------
            metrics_tracker.update_per_batch('train' if train else 'val', loss, task_losses_dict,
                                             pred_dict_for_loss, target_dict_for_loss,
                                             num_train_inputs_in_batch,
                                             pred_reposed_vertices=pred_reposed_vertices,
                                             target_reposed_vertices=target_reposed_vertices)

    def train(self):
        current_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in self.opts.save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(self.regressor.state_dict())

        # Instantiate metrics tracker.
        metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=self.opts.losses_to_track,
                                                          metrics_to_track=self.opts.metrics_to_track,
                                                          img_wh=self.opts.regressor_input_dim,
                                                          log_path=self.log_path,
                                                          load_logs=False,
                                                          current_epoch=current_epoch)

        # Starting training loop
        for epoch in range(current_epoch, self.opts.num_epochs):
            print('\nEpoch {}/{}'.format(epoch, self.opts.num_epochs - 1))
            print('-' * 10)
            metrics_tracker.initialise_loss_metric_sums()

            # Train
            self.regressor.train()
            self.run_epoch(self.train_dataloader, metrics_tracker, train=True)

            # Eval
            self.regressor.eval()
            with torch.no_grad():
                self.run_epoch(self.val_dataloader, metrics_tracker, train=False)

            metrics_tracker.update_per_epoch()

            save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(self.opts.save_val_metrics,
                                                                                                    best_epoch_val_metrics)

            if save_model_weights_this_epoch:
                for metric in self.opts.save_val_metrics:
                    best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
                print("Best epoch val metrics updated to ", best_epoch_val_metrics)
                best_model_wts = copy.deepcopy(self.regressor.state_dict())
                best_epoch = epoch
                print("Best model weights updated!")

            if epoch % self.opts.epochs_per_save == 0:
                # Saving current epoch num, best epoch num, best validation metrics (occurred in best
                # epoch num), current regressor state_dict, best regressor state_dict, current
                # optimiser state dict and current criterion state_dict (i.e. multi-task loss weights).
                save_dict = {'epoch': epoch,
                             'best_epoch': best_epoch,
                             'best_epoch_val_metrics': best_epoch_val_metrics,
                             'model_state_dict': self.regressor.state_dict(),
                             'best_model_state_dict': best_model_wts,
                             'optimiser_state_dict': self.optimiser.state_dict(),
                             'criterion_state_dict': self.criterion.state_dict()}
                torch.save(save_dict, self.model_save_path + '_epoch{}'.format(epoch) + '.tar')
                print('Model saved! Best Val Metrics:\n', best_epoch_val_metrics, '\nin epoch {}'.format(best_epoch))

            # TODO: this causes some segfault
            # predict_3D(os.path.join(os.path.dirname(__file__),'..', 'demo'),
            #            regressor.cpu(), torch.device("cpu"),
            #            silhouettes_from='sam',
            #            outpath=model_save_path + f'test_epoch{epoch}')
            # self.regressor.to(device)

        print('Training Completed. Best Val Metrics:\n',
              best_epoch_val_metrics)

        self.regressor.load_state_dict(best_model_wts)


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
