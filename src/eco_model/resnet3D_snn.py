"""
SNN版
Inception-v2で次元削減したフレームを、時間方向にも畳み込んで特徴量を絞り出す3DConv

SNN版のResNetの構造は↓この論文のMS ResNetってのを参考にしている.
https://arxiv.org/pdf/2112.08954.pdf (Fig2)
簡単な説明としては、スパイクを軸索＆シナプスでシナプス電流に変換したあとのものをResidualとして足し合わせる.
"""

import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

class ResNetSNN3D(nn.Module):
    def __init__(self,snn_threshold):
        super().__init__()
        
        self.resnet3d_3=ResnetSNN_3D_3(th=snn_threshold)
        self.resnet3d_4=ResnetSNN_3D_4(th=snn_threshold)
        self.resnet3d_5=ResnetSNN_3D_5(th=snn_threshold)
        
        #>> SNNの方でAvgPoolするのは良くない. spikeをpoolingしても1か0しか無いので、情報が消えるだけ >>
        # #avgPoolを重み固定の軸索＆シナプスとみなす
        # self.global_avg_pool=nn.AvgPool3d(
        #     kernel_size=(4,2,2),stride=1,padding=0
        # )
        # self.out_lif_neuron=snn.Leaky(
        #     beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th,
        #     threshold=1,output=True)
        #>> SNNの方でAvgPoolするのは良くない >>
        
    
    def forward(self,x:torch.Tensor):
        """
        :param x : [batch x channel x time-sequence x h x w]
        :return out : [batch x out_dim]
        """

        out=self.resnet3d_3(x)
        out=self.resnet3d_4(out)
        out=self.resnet3d_5(out)            

        #>> SNNの方でAvgPoolするのは良くない spikeをpoolingしても1か0しか無いので、情報が消えるだけ>>
        # out:torch.Tensor=self.global_avg_pool(out) #重み固定の軸索＆シナプスとみなす. avgPoolによってシナプス電流に変換するとみなす.
        # out=out.view(out.shape[0],out.shape[1]) #余計な次元を落とす
        # # print(f"3DCNN avgPool : {torch.sum(out)}")
        # out,out_mem=self.out_lif_neuron(out)
        #>> SNNの方でAvgPoolするのは良くない >>

        out=torch.flatten(out,start_dim=1)

        return out


class ResnetSNN_3D_3(nn.Module):
    '''Resnet_3D_3'''

    def __init__(self,th):
        super().__init__()
        
        # 軸索＆シナプス役　電流へ変換
        self.res3a_2 = nn.Conv3d(32, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        
        self.res3a_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        )

        self.res3b_1 = nn.Conv3d(64, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        )
        
        self.res3b_2 = nn.Conv3d(64, 64, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) #ここで電流へ変換
        
        self.res3b_bn = nn.BatchNorm3d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_2_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th,
            output=True
        ) #最後にスパイクへ

    def forward(self, x):
        """
        :param x : 1step分のspike [batch x channel x time-sequence x h x w]
        """

        residual = self.res3a_2(x) #シナプス電流へ変換
        
        out = self.res3a_bn(residual)
        out = self.res3a_1_snn(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_1_snn(out)
        out = self.res3b_2(out)

        out += residual #residualシナプス電流を足し合わせる

        out = self.res3b_bn(out)
        out,out_mem = self.res3b_2_snn(out) #最後にLIFにぶち込んでspikeにする

        return out
    
    
class ResnetSNN_3D_4(nn.Module):
    '''Resnet_3D_4'''

    def __init__(self,th):
        super().__init__()

        #>> Layer1 >>
        self.res4a_1 = nn.Conv3d(64, 128, kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0,0,0)) #シナプス電流へ変換
        self.res4a_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        self.res4a_2 = nn.Conv3d(128,128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) #シナプス電流へ
        
        self.res4a_down = nn.Conv3d(64, 128, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0)) #シナプス電流へ(residual)
        #>> Layer1 >>
        
        #>> Layer2 >>
        self.res4a_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_2_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        
        self.res4b_1 = nn.Conv3d(128, 128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) #シナプス電流へ
        self.res4b_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        self.res4b_2 = nn.Conv3d(128,128, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) #シナプス電流へ
        #>> Layer2 >>
        
        self.res4b_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.res4b_2_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th,
            output=True
        )

    def forward(self, x):
        residual = self.res4a_down(x) #residual シナプス電流

        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_snn(out)

        out = self.res4a_2(out)

        out += residual #シナプス電流足し合わせ

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_2_snn(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_snn(out)

        out = self.res4b_2(out)

        out += residual2 #シナプス電流を足し合わせ

        out = self.res4b_bn(out)
        out,out_mem = self.res4b_2_snn(out)

        return out
    
    
class ResnetSNN_3D_5(nn.Module):
    '''Resnet_3D_5
    '''

    def __init__(self,th):
        super().__init__()
        
        #>> Layer1 >>
        self.res5a_1 = nn.Conv3d(128,256, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0))
        self.res5a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        self.res5a_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        self.res5a_down = nn.Conv3d(128, 256, kernel_size=(
            2,2,2), stride=(2, 2, 2), padding=(0,0,0))
        #>> Layer1 >>
        
        
        #>> Layer2 >>
        self.res5a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5a_2_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        
        self.res5b_1 = nn.Conv3d(256,256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res5b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_1_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th
        ) #スパイクへ
        self.res5b_2 = nn.Conv3d(256,256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        #>> Layer2 >>
        
        self.res5b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res5b_2_snn=snn.Leaky(
            beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25), init_hidden=True, threshold=th,
            output=True
        ) #スパイクへ



    def forward(self, x):
     
        residual = self.res5a_down(x)

        out = self.res5a_1(x)
        out = self.res5a_1_bn(out)
        out = self.res5a_1_snn(out)

        out = self.res5a_2(out)

        out += residual  # res5a

        residual2 = out

        out = self.res5a_bn(out)
        out = self.res5a_1_snn(out)

        out = self.res5b_1(out)

        out = self.res5b_1_bn(out)
        out = self.res5b_1_snn(out)

        out = self.res5b_2(out)

        out += residual2  # res5b
        
        out = self.res5b_bn(out)
                
        out_sp,out_mem = self.res5b_2_snn(out)
        
        return out_sp