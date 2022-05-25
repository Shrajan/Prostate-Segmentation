# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : models.py
    Code Author : Shrajan Bhandary
    Created on  : 19 March 2021
    Program Description:
        This program fetches required model from the list of networks.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    19 March 2021      |     Shrajan Bhandary     |  Implemented necessary functions.   |
    |    16 March 2021      |     Shrajan Bhandary     |    Added V-Net model to the list.   |
    |    23 March 2021      |     Shrajan Bhandary     |   Added U-Net++ model to the list.  |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# GLOBAL VARIABLE: list of networks.

allModels = ["unet", "attention_unet", "segresnet", "vnet", "unet_pp"]

# IMPLEMENTATION

def get_model(opt=None):
    """
    Retrieves the model architecture based on the requirement.
    Args: model_name: Must be a string from choice list.
          opt: Command line argument.
    """
    # Original UNet
    if opt.model_name == "unet":
        from networks.unet import UNet as model_class

    # Attention-UNet
    elif opt.model_name == "attention_unet":
        from networks.attention_unet import Attention_UNet as model_class

    # Seg ResNet
    elif opt.model_name == "segresnet":
        from monai.networks.nets import SegResNet 
        model = SegResNet(in_channels=opt.input_channels, out_channels=opt.output_channels, init_filters=opt.n_kernels, dropout_prob=opt.dropout_rate)
        return model

    # V-Net
    elif opt.model_name == "vnet":
        from networks.vnet import VNet as model_class

    # U-Net++
    elif opt.model_name == "unet_pp":
        from networks.unet_pp import UNet_Plus_Plus as model_class

    else:
        raise Exception("Re-check the model name, otherwise the model isn't available.")
    
    model = model_class(opt)
    return model

