import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

from .inception_v2_nn import InceptionV2
from .resnet3D_nn import ResNet3D
from .inception_v2_snn import InceptionV2SNN
from .resnet3D_snn import ResNetSNN3D

class ECO(nn.Module):

    def __init__(self):
        super().__init__()
        self.inception_v2=InceptionV2()
        self.resnet3d=ResNet3D()
        self.out_layer=nn.Linear(
            in_features=4096,out_features=10,bias=True
        )

    def forward(self,x:torch.Tensor):
        """
        :param x [batch x time_sequence x channel x h x w]
        """

        n,t,c,h,w=x.shape
        x=x.view(n*t,c,h,w)

        out:torch.Tensor=self.inception_v2(x) #n*t x c x h x w
        _,c_out,h_out,w_out=out.shape
        out=out.view(n,t,c_out,h_out,w_out) #時系列を復活させる
        
        out=torch.transpose(out,dim0=1,dim1=2) #時間軸とchannel軸をいれかえる
        out=self.resnet3d(out)

        out=self.out_layer(out)

        return out


class ECOSNN(nn.Module):
    def __init__(self,snn_time_step=30,snn_threshold=0.3):
        super().__init__()

        self.snn_time_step=snn_time_step
        
        self.inception_v2=InceptionV2SNN(snn_threshold)
        self.resnet3d=ResNetSNN3D(snn_threshold)

        self.out_layer=nn.Sequential(
            nn.Linear(in_features=4096,out_features=10,bias=True),
            snn.Leaky(
                    beta=0.5,spike_grad=surrogate.fast_sigmoid(slope=25),
                    init_hidden=True,output=True,threshold=snn_threshold
                )
        )
        

    def encode(self,x:torch.Tensor):
        """
        アナログ入力をSNNの時間方向に引き伸ばす関数
        :param x [batch x channel x h x w]
        :return out [snn_time_steps x batch x channel x h x w]
        """

        n,c,h,w=x.shape
        # out=x.repeat(self.snn_time_step,1,1,1).view(self.snn_time_step,n,c,h,w) #direct input

        # >> poisson_encoder >>
        out=[torch.where(
            torch.rand(size=x.shape)<=x,
            1,
            0
        ) for _ in range(self.snn_time_step)]
        out=torch.stack(out).type(torch.float)

        # outの1バッチ目の平均を取る
        # import matplotlib.pyplot as plt
        # import numpy as np
        # batch1_mean = torch.mean(out[:, 13, :, :, :], dim=0)       
        # plt.imshow(np.fliplr(batch1_mean[0].cpu().detach().numpy()))  # CPUに移動し、NumPy配列に変換
        # plt.savefig('batch2_first_image.png')  # 画像を保存
        # exit(1)
        
        # >> poisson_encoder >>

        return out
    

    def forward(self,x:torch.Tensor):
        """
        :param x [batch x time_sequence x channel x h x w]
        """

        utils.reset(self.inception_v2)
        utils.reset(self.resnet3d)
        utils.reset(self.out_layer)

        n,t,c,h,w=x.shape
        x=x.view(n*t,c,h,w)

        out=self.encode(x) #[snn_time_step x n*t x c h x w] snnの時間方向を引き伸ばす

        out_sp=[]
        for step in range(self.snn_time_step):
  
            out_step=self.inception_v2.forward(out[step]) #[n*t x c x h x w]
            _,c_out,h_out,w_out=out_step.shape
            out_step=out_step.view(n,t,c_out,h_out,w_out) #時系列を復活させる

            out_step=torch.transpose(out_step,1,2)
            out_step=self.resnet3d.forward(out_step)

            out_step,_=self.out_layer(out_step)
            out_sp.append(out_step)

        return torch.stack(out_sp)        