"""
SNN版
２次元のフレーム一枚一枚を次元削減するネットワーク
"""

import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt


class InceptionV2SNN(nn.Module):
    def __init__(self):
        super(InceptionV2SNN, self).__init__()

        self.basic_conv = BasicConvSNN()
        self.inception_a = InceptionASNN()
        self.inception_b = InceptionBSNN()
        self.inception_c = InceptionCSNN()

    def forward(self, x:torch.Tensor):
        """
        :param x: [SNN-time-steps x batch x channel x h x w]
        :return out_sp: [SNN-time-steps x batch x channel x h x w]
        """
        out_sp=[]
        
        for t in range(x.shape[0]):

            out = self.basic_conv(x[t])
            out = self.inception_a(out)
            out = self.inception_b(out)
            out = self.inception_c(out)
            out_sp.append(out)
            
        return torch.stack(out_sp)


class BasicConvSNN(nn.Module):
    '''ECOの2D Netモジュールの最初のモジュール'''

    def __init__(self):
        super(BasicConvSNN, self).__init__()

        self.conv1_7x7_s2 = nn.Conv2d(1, 32, kernel_size=(
            6, 6), stride=(2, 2), padding=(2, 2))
        self.conv1_7x7_s2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.pool1_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.

        self.conv2_3x3_reduce = nn.Conv2d(
            32, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True
        )

        self.conv2_3x3 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2_3x3_s2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2_2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=True, output=True
        )

    def forward(self, x):
        out = self.conv1_7x7_s2(x)
        out = self.conv1_7x7_s2_bn(out)

        out = self.pool1_3x3_s2(out)
        out = self.conv1_snn(out)

        out = self.conv2_3x3_reduce(out)
        out = self.conv2_3x3_reduce_bn(out)
        out = self.conv2_1_snn(out)

        out = self.conv2_3x3(out)
        out = self.conv2_3x3_bn(out)
        out = self.pool2_3x3_s2(out)
        out, out_mem = self.conv2_2_snn(out)

        return out


class InceptionASNN(nn.Module):
    '''InceptionA'''

    def __init__(self):
        super(InceptionASNN, self).__init__()

        # >> Layer1 >>
        self.inception_3a_1x1 = nn.Conv2d(
            64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        # >> Layer1 >>

        # >> Layer2 >>
        self.inception_3a_3x3_reduce = nn.Conv2d(
            64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a2_1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True
        )

        self.inception_3a_3x3 = nn.Conv2d(
            16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a2_2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )
        # >> Layer2 >>

        # >> Layer3 >>
        self.inception_3a_double_3x3_reduce = nn.Conv2d(
            64, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a3_1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True
        )

        self.inception_3a_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a3_2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
        )

        self.inception_3a_double_3x3_2 = nn.Conv2d(
            32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a3_3_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )
        # >> Layer3 >>

        # >> Layer4 >>
        self.inception_3a_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3a_pool_proj = nn.Conv2d(
            64, 8, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = nn.BatchNorm2d(
            8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_a4_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )
        # >> Layer4 >>

    def forward(self, x):

        out1 = self.inception_3a_1x1(x)
        out1 = self.inception_3a_1x1_bn(out1)
        out1,out1_mem = self.inception_a1_snn(out1)

        out2 = self.inception_3a_3x3_reduce(x)
        out2 = self.inception_3a_3x3_reduce_bn(out2)
        out2 = self.inception_a2_1_snn(out2)
        out2 = self.inception_3a_3x3(out2)
        out2 = self.inception_3a_3x3_bn(out2)
        out2,out2_mem = self.inception_a2_2_snn(out2)

        out3 = self.inception_3a_double_3x3_reduce(x)
        out3 = self.inception_3a_double_3x3_reduce_bn(out3)
        out3 = self.inception_a3_1_snn(out3)
        out3 = self.inception_3a_double_3x3_1(out3)
        out3 = self.inception_3a_double_3x3_1_bn(out3)
        out3 = self.inception_a3_2_snn(out3)
        out3 = self.inception_3a_double_3x3_2(out3)
        out3 = self.inception_3a_double_3x3_2_bn(out3)
        out3,out3_mem = self.inception_a3_3_snn(out3)

        out4 = self.inception_3a_pool(x)
        out4 = self.inception_3a_pool_proj(out4)
        out4 = self.inception_3a_pool_proj_bn(out4)
        out4,out4_mem = self.inception_a4_snn(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)


class InceptionBSNN(nn.Module):
    '''InceptionB'''

    def __init__(self):
        super(InceptionBSNN, self).__init__()

        # >> Layer1 >>
        self.inception_3b_1x1 = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        
        # >> Layer1 >>

        # >> Layer2 >>
        self.inception_3b_3x3_reduce = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b2_1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.

        self.inception_3b_3x3 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b2_2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        # >> Layer2 >>

        # >> Layer3 >>
        self.inception_3b_double_3x3_reduce = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b3_1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.

        self.inception_3b_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b3_2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        
        self.inception_3b_double_3x3_2 = nn.Conv2d(
            32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b3_3_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        # >> Layer3 >>

        # >> Layer4 >>
        self.inception_3b_pool = nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1)
        self.inception_3b_pool_proj = nn.Conv2d(
            72, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_b4_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )  # SNNは基本的にpoolingの後ろにつけないと意味がない. なぜなら出力が0か1なので,poolingしたらだいたい1になるから.
        # >> Layer4 >>

    def forward(self, x):

        out1 = self.inception_3b_1x1(x)
        out1 = self.inception_3b_1x1_bn(out1)
        out1,out1_mem = self.inception_b1_snn(out1)

        out2 = self.inception_3b_3x3_reduce(x)
        out2 = self.inception_3b_3x3_reduce_bn(out2)
        out2 = self.inception_b2_1_snn(out2)
        out2 = self.inception_3b_3x3(out2)
        out2 = self.inception_3b_3x3_bn(out2)
        out2,out2_mem = self.inception_b2_2_snn(out2)

        out3 = self.inception_3b_double_3x3_reduce(x)
        out3 = self.inception_3b_double_3x3_reduce_bn(out3)
        out3 = self.inception_b3_1_snn(out3)
        out3 = self.inception_3b_double_3x3_1(out3)
        out3 = self.inception_3b_double_3x3_1_bn(out3)
        out3 = self.inception_b3_2_snn(out3)
        out3 = self.inception_3b_double_3x3_2(out3)
        out3 = self.inception_3b_double_3x3_2_bn(out3)
        out3,out3_mem = self.inception_b3_3_snn(out3)

        out4 = self.inception_3b_pool(x)
        out4 = self.inception_3b_pool_proj(out4)
        out4 = self.inception_3b_pool_proj_bn(out4)
        out4,out4_mem = self.inception_b4_snn(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)


class InceptionCSNN(nn.Module):
    '''InceptionC'''

    def __init__(self):
        super(InceptionCSNN, self).__init__()

        self.inception_3c_double_3x3_reduce = nn.Conv2d(
            96, 16, kernel_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_c1_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
        )

        self.inception_3c_double_3x3_1 = nn.Conv2d(
            16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inception_c2_snn = snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True,
            output=True
        )
    def forward(self, x):
        out = self.inception_3c_double_3x3_reduce(x)
        out = self.inception_3c_double_3x3_reduce_bn(out)
        out = self.inception_c1_snn(out)
        out = self.inception_3c_double_3x3_1(out)
        out = self.inception_3c_double_3x3_1_bn(out)
        out,out_mem = self.inception_c2_snn(out)

        return out
