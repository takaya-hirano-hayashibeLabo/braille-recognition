from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent
import sys
sys.path.append(str(ROOT))
import os

import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import matplotlib.pyplot as plt

def main():
    """
    snnTorchのSNNを3次元に出来るかやってみる.
    なんか、できそうな感じがするぞ！
    できそうなので、時間方向に畳み込むことで、時系列処理が出来るようになると思います.e
    """
    
    net3d=nn.Sequential(
        nn.Conv3d(in_channels=1,out_channels=4,kernel_size=5),
        nn.MaxPool3d(2),
        snn.Leaky(
            beta=0.5,spike_grad=surrogate.fast_sigmoid(slope=25),
            init_hidden=True,output=True
        )
    )
    
    #[channel x time sequence x H x W] = [1 x 16 x 64 x 64]
    input_data=torch.rand(size=(1,16,64,64)) 
    
    spike_out,mem_out=net3d(input_data)
    print(spike_out)
    print("-"*50)
    print(mem_out)
    
    
if __name__=="__main__":
    main()