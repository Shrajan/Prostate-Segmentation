allModels = ["unet", "attention_unet", "segresnet", "vnet", "unet_pp"]

def get_model(opt=None):
    """
    Retrieves the model architecture based on the requirement.
    Args: model_name: Must be a string from choice list.
          opt: Command line argument.
    """
    # Original UNet
    if opt.model_name == "unet":
        from networks.unet import UNet
        model = UNet(input_channels = opt.input_channels, 
                    output_channels = opt.output_channels, 
                    n_kernels = opt.n_kernels,
                    dropout_rate=opt.dropout_rate)
        return model

    # Attention-UNet
    elif opt.model_name == "attention_unet":
        from networks.attention_unet import Attention_UNet
        model = Attention_UNet(input_channels = opt.input_channels, 
                                output_channels = opt.output_channels, 
                                n_kernels = opt.n_kernels)
        return model
    
    # Seg ResNet
    elif opt.model_name == "segresnet":
        from monai.networks.nets import SegResNet 
        model = SegResNet(in_channels=opt.input_channels, out_channels=opt.output_channels, init_filters=opt.n_kernels, dropout_prob=opt.dropout_rate)
        return model

    # V-Net
    elif opt.model_name == "vnet":
        from networks.vnet import VNet
        model = VNet(input_channels = opt.input_channels, 
                    output_channels = opt.output_channels, 
                    n_kernels = opt.n_kernels,
                    dropout_rate=opt.dropout_rate) 
        return model

    # U-Net++
    elif opt.model_name == "unet_pp":
        from networks.unet_pp import UNet_Plus_Plus
        model = UNet_Plus_Plus(input_channels = opt.input_channels, 
                    output_channels = opt.output_channels, 
                    n_kernels = opt.n_kernels,
                    dropout_rate=opt.dropout_rate,
                    deep_supervision=False)
        return model

    else:
        raise Exception("Re-check the model name, otherwise the model isn't available.")
