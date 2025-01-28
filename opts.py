class Opts:
    experiment_name = 'test'
    num_workers = 4
    pin_memory = True
    device = 'cuda:0'

    # ----------------------- Network settings -----------------------
    resnet_in_channels = 1 + 17  # single-channel silhouette + 17 joint heatmaps
    resnet_layers = 18
    ief_iters = 3

    # ----------------------- Hyperparameters -----------------------
    num_epochs = 100
    batch_size = 100
    lr = 0.001
    epochs_per_save = 10

    # ----------------------- Data settings -----------------------
    val_perc = 0.001
    mean_camera_t = [0., 0.2, 42.]
    focal_length = 5000.
    regressor_input_dim = 256

    # ----------------------- Loss settings -----------------------
    losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
    init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                         'joints3D': 1.0}  # Initial loss weights - these will be updated during training.
    losses_to_track = losses_on
    normalise_joints_before_loss = True

    # ----------------------- Metrics settings -----------------------
    metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc',
                        'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
    save_val_metrics = ['pves_pa', 'mpjpes_pa']

    # ----------------------- Augmentations -----------------------
    augment_shape = True
    delta_betas_distribution = 'normal'
    import torch # TODO check if can be removed
    delta_betas_std = 1.5  # used if delta_betas_distribution is 'normal'
    delta_betas_range = [-3., 3.]  # used if delta_betas_distribution is 'uniform'
    smpl_augment_params = {'augment_shape': augment_shape,
                           'delta_betas_distribution': delta_betas_distribution,
                           'delta_betas_std': delta_betas_std,
                           'delta_betas_range': delta_betas_range}
    # Camera
    xy_std = 0.05
    delta_z_range = [-5, 5]
    cam_augment_params = {'xy_std': xy_std,
                          'delta_z_range': delta_z_range}
    # BBox
    crop_input = True # ARiel: shift and rescale augmentation
    mean_scale_factor = 1.2
    delta_scale_range = [-0.2, 0.2]
    delta_centre_range = [-5, 5]
    bbox_augment_params = {'crop_input': crop_input,
                           'mean_scale_factor': mean_scale_factor,
                           'delta_scale_range': delta_scale_range,
                           'delta_centre_range': delta_centre_range}
    # Proxy Representation
    condition_on_depth = True
    remove_appendages = False # Removes entire bodyparts
    deviate_joints2D = True
    deviate_verts2D = True # TODO Ariel I changed this to False
    occlude_seg = False

    remove_appendages_classes = [1, 2, 3, 4, 5, 6]
    remove_appendages_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    delta_j2d_dev_range = [-8, 8]
    delta_j2d_hip_dev_range = [-8, 8]
    delta_verts2d_dev_range = [-0.01, 0.01] # [-0.0025, 0.0025] # TODO Ariel I changed this from [-0.01, 0.01]
    occlude_probability = 0.5
    occlude_box_dim = 48

    proxy_rep_augment_params = {'remove_appendages': remove_appendages,
                                'deviate_joints2D': deviate_joints2D,
                                'deviate_verts2D': deviate_verts2D,
                                'occlude_seg': occlude_seg,
                                'remove_appendages_classes': remove_appendages_classes,
                                'remove_appendages_probabilities': remove_appendages_probabilities,
                                'delta_j2d_dev_range': delta_j2d_dev_range,
                                'delta_j2d_hip_dev_range': delta_j2d_hip_dev_range,
                                'delta_verts2d_dev_range': delta_verts2d_dev_range,
                                'occlude_probability': occlude_probability,
                                'occlude_box_dim': occlude_box_dim}