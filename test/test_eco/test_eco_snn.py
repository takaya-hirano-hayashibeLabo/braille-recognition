from pathlib import Path
ROOT=str(Path(__file__).parent.parent.parent)
import sys
sys.path.append(f"{ROOT}")

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as arani
import numpy as np

from src.eco_model.inception_v2_snn import *
from src.eco_model.resnet3D_snn import *

def show_spike(out,fignum=1):
    fig=plt.figure(fignum)
    frames=[[plt.imshow(frame,cmap="gray")] for frame in out[0].detach().to("cpu").numpy()]
    ani=arani(fig,artists=frames,interval=100)
    plt.show()

if __name__=="__main__":
    """
    【概要】
    Inception-v2の出力次元テスト
    ResNet3Dの出力次元テスト
    
    【結果】
    OK. 設計通り.
    """
    
    #> Inception-v2のテスト >
    test_data=torch.rand(size=(1,1,64,64))
    
    #>> ネットワークの準備 >>
    basic_conv=BasicConvSNN()
    inceptionA=InceptionASNN()
    inceptionB=InceptionBSNN()
    inceptionC=InceptionCSNN()
    
    inception_v2=InceptionV2SNN()
    #>> ネットワークの準備 >>
    
    
    #>> ネットワークに通す >>
    print("[BasicConv]"+"="*50)
    out:torch.Tensor=basic_conv.forward(test_data)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 64 x 8 x 8]\n")
    
    print("[InceptionA]"+"="*50)
    out:torch.Tensor=inceptionA.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 72 x 8 x 8]\n")
    # show_spike(out)
    
    print("[InceptionB]"+"="*50)
    out:torch.Tensor=inceptionB.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 96 x 8 x 8]\n")
    # show_spike(out)
    
    print("[InceptionC]"+"="*50)
    out:torch.Tensor=inceptionC.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 32 x 8 x 8]\n")
    
    print("[InceptionV2]"+"="*50)
    out:torch.Tensor=inception_v2.forward(x=torch.rand(size=(16,1,1,64,64)))
    print(f"out dim : {out.shape}")
    print(f"exp dim :[16 x 1 x 32 x 8 x 8]\n")
    # show_spike(torch.transpose(out[:,:,0],1,0))
    #>> ネットワークに通す >>
    
    #> Inception-v2のテスト >
    
    

    #> ResNet3Dのテスト >>
    test_data=torch.where(
        torch.rand(size=(1,32,16,8,8))>0.2,
        1.0,0.0
    )
    
    #>> ネットワーク >>
    res3=ResnetSNN_3D_3()
    res4=ResnetSNN_3D_4()
    res5=ResnetSNN_3D_5()
    
    resnet3d=ResNetSNN3D()
    
    #>> ネットワークに通す >>
    print("[ResNet_3]"+"="*50)
    out:torch.Tensor=res3.forward(test_data)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 64 x 16 x 8 x 8]\n")
    # show_spike((out[:,0]))
    
    print("[ResNet_4]"+"="*50)
    out:torch.Tensor=res4.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 128 x 8 x 4 x 4]\n")
    # show_spike((out[:,0]),fignum=2)
    print(torch.sum(out))
    
    
    print("[ResNet_5]"+"="*50)
    out:torch.Tensor=res5.forward(out)
    # print(f"out dim : {out.shape}")
    # print(f"exp dim :[1 x 256 x 4 x 2 x 2]\n")
    # print(torch.sum(out))
    # # show_spike((out[:,0]),fignum=3)
    
    
    print("[ResNet3D]"+"="*50)
    snn_time_steps=30
    out=resnet3d.forward(
        torch.where(
            torch.rand(size=(snn_time_steps,1,32,16,8,8))>0.1,
            1.0,0.0
        )
    )
    print(f"out dim : {out.shape}")
    print(f"exp dim :[{snn_time_steps} x 1 x 256]\n")
    
    out=out.view((-1,snn_time_steps)).detach().to("cpu").numpy()
    print(np.sum(out))
    spike_idx=np.where(out==1)
    # print(spike_idx)
    fig=plt.figure(100)
    
    # for i in range(5):
    #     out_mem_i=out_mems[:,0,i].to("cpu").detach().numpy()
    #     plt.plot(out_mem_i,alpha=0.7)
    # plt.show()
    
    plt.plot(spike_idx[1],spike_idx[0],".")
    plt.show()
    #>> ネットワークに通す >>
    