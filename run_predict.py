import os
import argparse
import torch

from models.regressor import SingleInputRegressor
from opts import Opts
from predict.predict_3D import predict_3D




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input image/folder of images.')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--silh_from', choices=['detectron2', 'sam'], default='sam')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    opts = Opts() # TODO read from ckpt folder

    regressor = SingleInputRegressor(resnet_in_channels=18,
                                     resnet_layers=18,
                                     ief_iters=3)

    print("Regressor loaded. Weights from:", args.checkpoint)
    regressor.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    regressor.load_state_dict(checkpoint['model_state_dict'])

    predict_3D(opts, args.input, regressor, device,
               silhouettes_from=args.silh_from,
               save_proxy_vis=True,
               render_vis=True,
               outpath=os.path.join(os.path.dirname(args.input), "..", "inference"))

