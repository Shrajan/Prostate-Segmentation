# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : unet_pp.py
    Code Author : Shrajan Bhandary
    Created on  : 23 March 2021
    Program Description:
        This program was forked from https://github.com/4uiiurz1/pytorch-nested-unet. It is
        based on the paper "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." 
        Article available at https://arxiv.org/abs/1807.10165.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|  
'''

# LIBRARY IMPORTS
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

# IMPLEMENTATION

class Conv_3x3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class UNet_Plus_Plus(nn.Module):
    def __init__(self, opt, deep_supervision=False):
        super().__init__()
        self.input_channels = opt.input_channels
        self.output_channels = opt.output_channels
        self.n_kernels = opt.n_kernels
        self.dropout_rate = opt.dropout_rate
        self.deep_supervision = deep_supervision

        filters = [self.n_kernels, self.n_kernels*2, self.n_kernels*4, self.n_kernels*8, self.n_kernels*16]
        self.pool_hwd = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool_hw = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.up_hwd = nn.Upsample(scale_factor=2, mode= "trilinear", align_corners=True)
        self.up_hw = nn.Upsample(scale_factor=(2, 2, 1), mode= "trilinear", align_corners=True)

        self.conv0_0 = Conv_3x3(self.input_channels, filters[0], filters[0])
        self.conv1_0 = Conv_3x3(filters[0], filters[1], filters[1])
        self.conv2_0 = Conv_3x3(filters[1], filters[2], filters[2])
        self.conv3_0 = Conv_3x3(filters[2], filters[3], filters[3])
        self.conv4_0 = Conv_3x3(filters[3], filters[4], filters[4])

        self.conv0_1 = Conv_3x3(filters[0]+filters[1], filters[0], filters[0])
        self.conv1_1 = Conv_3x3(filters[1]+filters[2], filters[1], filters[1])
        self.conv2_1 = Conv_3x3(filters[2]+filters[3], filters[2], filters[2])
        self.conv3_1 = Conv_3x3(filters[3]+filters[4], filters[3], filters[3])

        self.conv0_2 = Conv_3x3(filters[0]*2+filters[1], filters[0], filters[0])
        self.conv1_2 = Conv_3x3(filters[1]*2+filters[2], filters[1], filters[1])
        self.conv2_2 = Conv_3x3(filters[2]*2+filters[3], filters[2], filters[2])

        self.conv0_3 = Conv_3x3(filters[0]*3+filters[1], filters[0], filters[0])
        self.conv1_3 = Conv_3x3(filters[1]*3+filters[2], filters[1], filters[1])

        self.conv0_4 = Conv_3x3(filters[0]*4+filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv3d(filters[0], self.output_channels, kernel_size=1)
            self.final2 = nn.Conv3d(filters[0], self.output_channels, kernel_size=1)
            self.final3 = nn.Conv3d(filters[0], self.output_channels, kernel_size=1)
            self.final4 = nn.Conv3d(filters[0], self.output_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(filters[0], self.output_channels, kernel_size=1)
        
        self.weight_init()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool_hwd(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_hwd(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool_hwd(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_hwd(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_hwd(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool_hw(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_hw(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_hwd(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_hwd(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool_hw(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up_hw(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up_hw(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up_hwd(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_hwd(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
    
    def weight_init(self):
          self.param_count_G = 0
          for module in self.modules():
            if (isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear)):
              init.orthogonal_(module.weight)
              self.param_count_G += sum([p.data.nelement() for p in module.parameters()])
          print("{} params initialized for model.".format(self.param_count_G))
