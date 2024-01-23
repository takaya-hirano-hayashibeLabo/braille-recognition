from pathlib import Path
ROOT=str(Path(__file__).parent.parent.parent)
import sys
sys.path.append(f"{ROOT}")

import torch

from src.eco_model.inception_v2_nn import *
from src.eco_model.resnet3D_nn import *

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
    basic_conv=BasicConv()
    inceptionA=InceptionA()
    inceptionB=InceptionB()
    inceptionC=InceptionC()
    
    inception_v2=InceptionV2()
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
    
    print("[InceptionB]"+"="*50)
    out:torch.Tensor=inceptionB.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 96 x 8 x 8]\n")
    
    print("[InceptionC]"+"="*50)
    out:torch.Tensor=inceptionC.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 32 x 8 x 8]\n")
    
    print("[InceptionV2]"+"="*50)
    out:torch.Tensor=inception_v2.forward(x=test_data)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 32 x 8 x 8]\n")
    #>> ネットワークに通す >>
    
    #> Inception-v2のテスト >
    
    

    #> ResNet3Dのテスト >>
    test_data=torch.rand(size=(1,32,16,8,8))
    
    #>> ネットワーク >>
    res3=Resnet_3D_3()
    res4=Resnet_3D_4()
    res5=Resnet_3D_5()
    
    resnet3d=ResNet3D()
    
    #>> ネットワークに通す >>
    print("[ResNet_3]"+"="*50)
    out:torch.Tensor=res3.forward(test_data)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 64 x 16 x 8 x 8]\n")
    
    print("[ResNet_4]"+"="*50)
    out:torch.Tensor=res4.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 128 x 8 x 4 x 4]\n")
    
    print("[ResNet_5]"+"="*50)
    out:torch.Tensor=res5.forward(out)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 256 x 4 x 2 x 2]\n")
    
    print("[ResNet3D]"+"="*50)
    out:torch.Tensor=resnet3d.forward(test_data)
    print(f"out dim : {out.shape}")
    print(f"exp dim :[1 x 256]\n")
    #>> ネットワークに通す >>