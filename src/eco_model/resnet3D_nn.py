"""
Inception-v2で次元削減したフレームを、時間方向にも畳み込んで特徴量を絞り出す3DConv
"""

import torch
from torch import nn


class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D,self).__init__()
        
        self.resnet3d_3=Resnet_3D_3()
        self.resnet3d_4=Resnet_3D_4()
        self.resnet3d_5=Resnet_3D_5()
        
        # self.global_avg_pool=nn.AvgPool3d(
        #     kernel_size=(4,2,2),stride=1,padding=0
        # )
        
    def forward(self,x):
        """
        :param x : [batch x channel x time-sequence x h x w]
        :return out : [batch x out_dim]
        """
        out=self.resnet3d_3(x)
        out=self.resnet3d_4(out)
        out=self.resnet3d_5(out)
        # out:torch.Tensor=self.global_avg_pool(out)
        
        # out=out.view(out.shape[0],out.shape[1]) #余計な次元を落とす
        out=torch.flatten(out,start_dim=1)
        
        return out


class Resnet_3D_3(nn.Module):
    '''Resnet_3D_3'''

    def __init__(self):
        super(Resnet_3D_3, self).__init__()
        
        self.res3a_2 = nn.Conv3d(32, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3a_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(64, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(64, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res3b_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.res3a_2(x)
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_relu(out)
        out = self.res3b_2(out)

        out += residual

        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out
    
    
class Resnet_3D_4(nn.Module):
    '''Resnet_3D_4'''

    def __init__(self):
        super(Resnet_3D_4, self).__init__()

        #>> Layer1 >>
        self.res4a_1 = nn.Conv3d(64, 128, kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0,0,0))
        self.res4a_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_relu = nn.ReLU(inplace=True)
        self.res4a_2 = nn.Conv3d(128,128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res4a_down = nn.Conv3d(64, 128, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0))
        #>> Layer1 >>
        
        #>> Layer2 >>
        self.res4a_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_relu = nn.ReLU(inplace=True)
        
        self.res4b_1 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res4b_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(128,128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #>> Layer2 >>
        
        self.res4b_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res4a_down(x)

        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_relu(out)

        out = self.res4a_2(out)

        out += residual

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_relu(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_relu(out)

        out = self.res4b_2(out)

        out += residual2

        out = self.res4b_bn(out)
        out = self.res4b_relu(out)

        return out
    
    
class Resnet_3D_5(nn.Module):
    '''Resnet_3D_5'''

    def __init__(self):
        super(Resnet_3D_5, self).__init__()
        
        #>> Layer1 >>
        self.res5a_1 = nn.Conv3d(128,256, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0))
        self.res5a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_1_relu = nn.ReLU(inplace=True)
        self.res5a_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5a_down = nn.Conv3d(128, 256, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0))
        #>> Layer1 >>
        
        
        #>> Layer2 >>
        self.res5a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_relu = nn.ReLU(inplace=True)
        
        self.res5b_1 = nn.Conv3d(256,256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res5b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_1_relu = nn.ReLU(inplace=True)
        self.res5b_2 = nn.Conv3d(256,256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #>> Layer2 >>
        
        self.res5b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res5a_down(x)

        out = self.res5a_1(x)
        out = self.res5a_1_bn(out)
        out = self.res5a_1_relu(out)

        out = self.res5a_2(out)

        out += residual  # res5a

        residual2 = out

        out = self.res5a_bn(out)
        out = self.res5a_relu(out)

        out = self.res5b_1(out)

        out = self.res5b_1_bn(out)
        out = self.res5b_1_relu(out)

        out = self.res5b_2(out)

        out += residual2  # res5b

        out = self.res5b_bn(out)
        out = self.res5b_relu(out)

        return out