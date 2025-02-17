import torch
import torch.nn as nn

from models.resnet import resnet18, resnet50
from models.ief_module import IEFModule


class SingleInputRegressor(nn.Module):
    """
    Combined encoder + regressor model that takes proxy representation input (e.g.
    silhouettes + 2D joints) and outputs SMPL body model parameters + weak-perspective
    camera.
    """
    def __init__(self,
                 resnet_in_channels=1,
                 resnet_layers=18,
                 ief_iters=3,
                 gender_dim=4):
        """
        :param resnet_in_channels: 1 if input silhouette/segmentation, 1 + num_joints if
        input silhouette/segmentation + joints.
        :param resnet_layers: number of layers in ResNet backbone (18 or 50)
        :param ief_iters: number of IEF iterations.
        """
        super(SingleInputRegressor, self).__init__()

        num_pose_params = 24*6
        num_output_params = 3 + num_pose_params + 10

        if resnet_layers == 18:
            self.image_encoder = resnet18(in_channels=resnet_in_channels,
                                          pretrained=False)
            self.ief_module = IEFModule([512, 512],
                                        512 + gender_dim,
                                        num_output_params,
                                        iterations=ief_iters)
            self.gender_embedding = nn.Embedding(2, gender_dim)  # 2 classes: 0 and 1

        elif resnet_layers == 50:
            self.image_encoder = resnet50(in_channels=resnet_in_channels,
                                          pretrained=False)
            self.ief_module = IEFModule([1024, 1024],
                                        2048 + gender_dim,
                                        num_output_params,
                                        iterations=ief_iters)
            self.gender_embedding = nn.Embedding(2, gender_dim)  # 2 classes: 0 and 1

    def forward(self, input, gender=0):
        input_feats = self.image_encoder(input)
        input_feats = torch.cat((input_feats, self.gender_embedding(gender)), dim=1)
        cam_params, pose_params, shape_params = self.ief_module(input_feats)

        return cam_params, pose_params, shape_params
