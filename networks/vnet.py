# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : vnet.py
    Code Author : Shrajan Bhandary
    Created on  : 16 March 2021
    Program Description:
        This program was forked from https://github.com/black0017/MedicalZooPytorch/blob/master/lib/medzoo/Vnet.py. 
        It is based on the paper "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
        Segmentation." Article available at https://arxiv.org/abs/1606.04797.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|  
'''

# LIBRARY IMPORTS
import torch.nn as nn
import torch
from torch.nn import init

# IMPLEMENTATION

def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition_HWD(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=None):
        super(DownTransition_HWD, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.dropout = nn.Dropout3d(dropout) if dropout is not None else None
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

class DownTransition_HW(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=None):
        super(DownTransition_HW, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=(2,2,1), stride=(2,2,1))
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.dropout = nn.Dropout3d(dropout) if dropout is not None else None
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out

class UpTransition_HWD(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=None):
        super(UpTransition_HWD, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do2 = nn.Dropout3d(0.5)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.dropout = nn.Dropout3d(dropout) if dropout is not None else None
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class UpTransition_HW(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=None):
        super(UpTransition_HW, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=(2,2,1), stride=(2,2,1))

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do2 = nn.Dropout3d(0.5)
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        self.dropout = nn.Dropout3d(dropout) if dropout is not None else None
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, input_channels = 1, output_channels = 1, n_kernels = 32, dropout_rate=0.25):
        super(VNet, self).__init__()
        self.elu = True
        self.output_channels = output_channels
        self.n_kernels = n_kernels
        self.dropout_rate = dropout_rate
        self.in_channels = input_channels

        self.in_tr = InputTransition(self.in_channels, elu=self.elu)
        self.down_tr32 = DownTransition_HWD(16, 2, self.elu)
        self.down_tr64 = DownTransition_HWD(32, 3, self.elu)
        self.down_tr128 = DownTransition_HW(64, 3, self.elu, dropout=self.dropout_rate)
        self.down_tr256 = DownTransition_HW(128, 3, self.elu, dropout=self.dropout_rate)
        self.up_tr256 = UpTransition_HW(256, 256, 3, self.elu, dropout=self.dropout_rate)
        self.up_tr128 = UpTransition_HW(256, 128, 3, self.elu, dropout=self.dropout_rate)
        self.up_tr64 = UpTransition_HWD(128, 64, 2, self.elu)
        self.up_tr32 = UpTransition_HWD(64, 32, 1, self.elu)
        self.out_tr = OutputTransition(32, self.output_channels, self.elu)
        self.weight_init()

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def weight_init(self):
        self.param_count_G = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)
                self.param_count_G += sum([p.data.nelement() for p in module.parameters()])
        print("{} params initialized for model.".format(self.param_count_G))

