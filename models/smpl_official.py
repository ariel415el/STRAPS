import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

import consts


class SMPL(torch.nn.Module):
    """
    Extension of the official SMPL (from the smplx python package) implementation to
    support more joints.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # super(SMPL, self).__init__(*args, **kwargs)
        self.male_model = _SMPL(*args, **kwargs, gender='male')
        self.female_model = _SMPL(*args, **kwargs, gender='female')
        J_regressor_extra = np.load(consts.J_REGRESSOR_EXTRA_PATH)
        J_regressor_cocoplus = np.load(consts.COCOPLUS_REGRESSOR_PATH)
        J_regressor_h36m = np.load(consts.H36M_REGRESSOR_PATH)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra,
                                                               dtype=torch.float32))
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus,
                                                                  dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m,
                                                              dtype=torch.float32))


    def forward(self, gender, *args, **kwargs):
        kwargs['get_skin'] = True
        # smpl_output = super(SMPL, self).forward(*args, **kwargs)
        male_smpl_output = self.male_model.forward(*args, **kwargs)
        female_smpl_output = self.female_model.forward(*args, **kwargs)

        vertices = torch.stack((male_smpl_output.vertices, female_smpl_output.vertices), dim=0)[gender, torch.arange(len(gender))]
        global_orient = torch.stack((male_smpl_output.global_orient, female_smpl_output.global_orient), dim=0)[gender, torch.arange(len(gender))]
        joints = torch.stack((male_smpl_output.joints, female_smpl_output.joints), dim=0)[gender, torch.arange(len(gender))]
        body_pose = torch.stack((male_smpl_output.body_pose, female_smpl_output.body_pose), dim=0)[gender, torch.arange(len(gender))]
        betas = torch.stack((male_smpl_output.betas, female_smpl_output.betas), dim=0)[gender, torch.arange(len(gender))]
        assert(male_smpl_output.full_pose is None)
        extra_joints = vertices2joints(self.J_regressor_extra, vertices)
        cocoplus_joints = vertices2joints(self.J_regressor_cocoplus, vertices)
        h36m_joints = vertices2joints(self.J_regressor_h36m, vertices)

        all_joints = torch.cat([joints, extra_joints, cocoplus_joints,
                                h36m_joints], dim=1)
        output = SMPLOutput(vertices=vertices,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             joints=all_joints,
                             betas=betas,
                             full_pose=None)
        return output
