"""
２次元のフレーム一枚一枚を次元削減するネットワーク
"""

import torch
from torch import nn

class InceptionV2(nn.Module):
    def __init__(self):
        super(InceptionV2,self).__init__()
        
        # >> 深いInceptionv2 >>
        # self.basic_conv=BasicConv()
        # self.inception_a=InceptionA()
        # self.inception_b=InceptionB()
        # self.inception_c=InceptionC()
        # >> 深いInceptionv2 >>
        

        #>> 簡易的なConv2D >>
        self.net=nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32,eps=1e-5,),
            nn.AvgPool2d(2),
            nn.ReLU(),
            )
        #>> 簡易的なConv2D >>
        
        
    def forward(self, x):
        """
        2次元のデータを圧縮するCNN
        :param x [batch x channel x h x w]
        :return out [batch x channel x h x w]
        """

        # >> 深いInceptionv2 >>
        # out=self.basic_conv(x)
        # out=self.inception_a(out)
        # out=self.inception_b(out)
        # out=self.inception_c(out)
        # >> 深いInceptionv2 >>


        #>> 簡易的なConv2D >>
        out=self.net(x)
        #>> 簡易的なConv2D >>

        return out


class BasicConv(nn.Module):
    '''ECOの2D Netモジュールの最初のモジュール'''

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(1, 32, kernel_size=(
            6, 6), stride=(2, 2), padding=(2, 2))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu_7x7 = nn.ReLU(inplace=True)
        
        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        
        self.conv2_3x3_reduce = nn.Conv2d(
            32, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3_reduce = nn.ReLU(inplace=True)
        
        self.conv2_3x3 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_relu_3x3 = nn.ReLU(inplace=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)

    def forward(self, x):
        out = self.conv1_7x7_s2(x)
        out = self.conv1_7x7_s2_bn(out)
        out = self.conv1_relu_7x7(out)
        out = self.pool1_3x3_s2(out)
        out = self.conv2_3x3_reduce(out)
        out = self.conv2_3x3_reduce_bn(out)
        out = self.conv2_relu_3x3_reduce(out)
        out = self.conv2_3x3(out)
        out = self.conv2_3x3_bn(out)
        out = self.conv2_relu_3x3(out)
        out = self.pool2_3x3_s2(out)
        return out    
    

class InceptionA(nn.Module):
    '''InceptionA'''

    def __init__(self):
        super(InceptionA, self).__init__()

        #>> Layer1 >>
        self.inception_3a_1x1 = nn.Conv2d(
            64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_1x1 = nn.ReLU(inplace=True)
        #>> Layer1 >>

        #>> Layer2 >>
        self.inception_3a_3x3_reduce = nn.Conv2d(
            64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3_reduce = nn.ReLU(inplace=True)
        
        self.inception_3a_3x3 = nn.Conv2d(
            16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_3x3 = nn.ReLU(inplace=True)
        #>> Layer2 >>
        
        #>> Layer3 >>
        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            64,16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        
        self.inception_3a_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_1 = nn.ReLU(inplace=True)
        
        self.inception_3a_double_3x3_2 = nn.Conv2d(
            32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_double_3x3_2 = nn.ReLU(inplace=True)
        #>> Layer3 >>
        
        #>> Layer4 >>
        self.inception_3a_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3a_pool_proj = nn.Conv2d(
            64, 8, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(
            8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3a_relu_pool_proj = nn.ReLU(inplace=True)
        #>> Layer4 >>

    def forward(self, x):

        out1 = self.inception_3a_1x1(x)
        out1 = self.inception_3a_1x1_bn(out1)
        out1 = self.inception_3a_relu_1x1(out1)

        out2 = self.inception_3a_3x3_reduce(x)
        out2 = self.inception_3a_3x3_reduce_bn(out2)
        out2 = self.inception_3a_relu_3x3_reduce(out2)
        out2 = self.inception_3a_3x3(out2)
        out2 = self.inception_3a_3x3_bn(out2)
        out2 = self.inception_3a_relu_3x3(out2)

        out3 = self.inception_3a_double_3x3_reduce(x)
        out3 = self.inception_3a_double_3x3_reduce_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_reduce(out3)
        out3 = self.inception_3a_double_3x3_1(out3)
        out3 = self.inception_3a_double_3x3_1_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_1(out3)
        out3 = self.inception_3a_double_3x3_2(out3)
        out3 = self.inception_3a_double_3x3_2_bn(out3)
        out3 = self.inception_3a_relu_double_3x3_2(out3)

        out4 = self.inception_3a_pool(x)
        out4 = self.inception_3a_pool_proj(out4)
        out4 = self.inception_3a_pool_proj_bn(out4)
        out4 = self.inception_3a_relu_pool_proj(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)
    
class InceptionB(nn.Module):
    '''InceptionB'''

    def __init__(self):
        super(InceptionB, self).__init__()
        
        #>> Layer1 >>
        self.inception_3b_1x1 = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_1x1 = nn.ReLU(inplace=True)
        #>> Layer1 >>

        #>> Layer2 >>
        self.inception_3b_3x3_reduce = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_3x3_reduce = nn.ReLU(inplace=True)
        
        self.inception_3b_3x3 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_3x3 = nn.ReLU(inplace=True)
        #>> Layer2 >>

        #>> Layer3 >>
        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        
        self.inception_3b_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_1 = nn.ReLU(inplace=True)
        
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_double_3x3_2 = nn.ReLU(inplace=True)
        #>> Layer3 >>

        #>> Layer4 >>
        self.inception_3b_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3b_pool_proj = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3b_relu_pool_proj = nn.ReLU(inplace=True)
        #>> Layer4 >>
        

    def forward(self, x):
        
        out1 = self.inception_3b_1x1(x)
        out1 = self.inception_3b_1x1_bn(out1)
        out1 = self.inception_3b_relu_1x1(out1)

        out2 = self.inception_3b_3x3_reduce(x)
        out2 = self.inception_3b_3x3_reduce_bn(out2)
        out2 = self.inception_3b_relu_3x3_reduce(out2)
        out2 = self.inception_3b_3x3(out2)
        out2 = self.inception_3b_3x3_bn(out2)
        out2 = self.inception_3b_relu_3x3(out2)

        out3 = self.inception_3b_double_3x3_reduce(x)
        out3 = self.inception_3b_double_3x3_reduce_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_reduce(out3)
        out3 = self.inception_3b_double_3x3_1(out3)
        out3 = self.inception_3b_double_3x3_1_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_1(out3)
        out3 = self.inception_3b_double_3x3_2(out3)
        out3 = self.inception_3b_double_3x3_2_bn(out3)
        out3 = self.inception_3b_relu_double_3x3_2(out3)

        out4 = self.inception_3b_pool(x)
        out4 = self.inception_3b_pool_proj(out4)
        out4 = self.inception_3b_pool_proj_bn(out4)
        out4 = self.inception_3b_relu_pool_proj(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)
    
class InceptionC(nn.Module):
    '''InceptionC'''

    def __init__(self):
        super(InceptionC, self).__init__()

        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            96, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3c_relu_double_3x3_reduce = nn.ReLU(inplace=True)
        
        self.inception_3c_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_3c_relu_double_3x3_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.inception_3c_double_3x3_reduce(x)
        out = self.inception_3c_double_3x3_reduce_bn(out)
        out = self.inception_3c_relu_double_3x3_reduce(out)
        out = self.inception_3c_double_3x3_1(out)
        out = self.inception_3c_double_3x3_1_bn(out)
        out = self.inception_3c_relu_double_3x3_1(out)

        return out