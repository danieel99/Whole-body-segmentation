import os
import sys
import torch 
import pandas as pd
import torch.utils.tensorboard as tb

from monai.networks.nets import UNet 
from monai.losses import DiceLoss
from monai.losses import DiceFocalLoss
from monai.networks.layers import Norm
from baselines.seg_baseline import SegBaseline
from torch.utils.tensorboard import SummaryWriter

class UnetMonai():
    """
    Class with implementation of methods needed to perform segmentation using built-in Unet Monai model.

    Args:
        unet_params (dict): parameters of monai unet
        save_path (str): path to save model
        loss_function (str): only 'dice' avaiable for now
        optimizer (str): only 'adam' available for now
        epochs (int): number of epochs to train model 
    """
    def __init__(self, unet_params: dict, save_path: str, loss_function: str='dice', optimizer: str='adam', epochs: int=500) -> None:
        super().__init__()
        self.model = UNet(
            spatial_dims=unet_params['spatial_dims'],       
            in_channels=unet_params['in_channels'],         
            out_channels=unet_params['out_channels'],       
            channels=unet_params['channels'],               
            strides=unet_params['strides'],                 
            num_res_units=unet_params['num_res_units'],     
            norm=unet_params['norm'],   
        )
        self.unet_params = unet_params
        self.save_path = save_path
        self.epochs = epochs
        
        if loss_function=='dice':
            self.loss_function = DiceFocalLoss(sigmoid=True)
        if optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def run_UnetMonai(self, root_path: str, csv_folder: str, name: str, checkpoint=None) -> None:
        """
        Function starting segmentation training with unet monai model.

        Args:
            root_path (str): path to directory with csv files.
        """
        segmentation = SegBaseline(root_path, csv_folder)
        writer = tb.SummaryWriter('path', flush_secs=1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25, eta_min=1e-6)
        name = name
        segmentation.training(self.model, self.save_path, self.loss_function, self.optimizer, self.epochs, name, writer, scheduler=scheduler, checkpoint=checkpoint)

if __name__=='__main__':
    root_path = os.path.abspath(os.getcwd())
    unet_params = {'spatial_dims': 3,
                    'in_channels': 2,
                    'out_channels': 1,
                    'channels': (32, 64, 128, 256, 512),
                    'strides': (2, 2, 2, 2),
                    'num_res_units': 2,
                    'norm': Norm.BATCH}

    name = 'name'
    csv_folder = 'csv_folder'
    output_models_file = 'output_models_file'

    # first training
    unet_monai = UnetMonai(unet_params, os.path.join('path', output_models_file))
    unet_monai.run_UnetMonai('path', csv_folder, name)

    ## continue training
    # path = 'path'
    # checkpoint = torch.load(path, map_location=torch.device('cuda:0'))

    # unet_monai = UnetMonai(unet_params, os.path.join('path', output_models_file))
    # unet_monai.run_UnetMonai(os.getcwd(), csv_folder, name, checkpoint)