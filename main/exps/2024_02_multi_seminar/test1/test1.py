from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent.parent.parent
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

# import snntorch as snn
# from snntorch import surrogate
# from snntorch import backprop
# from snntorch import functional as SF
# from snntorch import utils
# from snntorch import spikeplot as splt

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as arani
from matplotlib import gridspec
from matplotlib.colors import Normalize # Normalizeをimport
from tqdm import tqdm
import pandas as pd

from src import ECO,ECOSNN


class Datasets(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

        self.datanum=x.shape[0]

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

class DataTransformStd():
    """
    ２次元データ（画像と同じ次元）をリサイズ＆標準化するクラス
    """
        
    def __call__(self,data,size=(28,28),mean=None,std=None):
        """
        :param data: [N x C x H x W]
        :param size: 変換後のサイズ
        :return data_nrm, mean, std
        """
        
        if not torch.is_tensor(data):
            data=torch.Tensor(data)
        
        if mean is None and std is None:
            mean=torch.mean(data)
            std=torch.std(data)
            
        data_nrm=F.interpolate(
            torch.Tensor((data-mean)/(1e-20+std)),
            size,mode='area'
        )
        
        return data_nrm,mean,std


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",required=True)
    args=parser.parse_args()

    input_data:np.ndarray=np.load(f"{args.data_dir}/input_3d.npy")
    input_data=input_data[:,:,np.newaxis,:,:] #channel方向に次元を伸ばす
    label_data:np.ndarray=np.load(f"{args.data_dir}/label.npy").astype(int)
    data_size=(64,64) #28, 64
    transform=DataTransformStd()
    n,t,c,h,w=input_data.shape
    input_data_nrm,mean,std=transform(input_data.reshape(n*t,c,h,w),data_size)
    input_data_nrm=input_data_nrm.view(n,t,c,data_size[0],data_size[1])
    # >> データのリサイズと標準化 >>

    # input_data_nrmの1次元の1つめ、2次元の1つ目のデータをflattenしてヒストグラムにする
    data_to_plot = input_data[:,0].flatten()
    print(data_to_plot)
    print(np.max(data_to_plot))
    plt.hist(data_to_plot[data_to_plot>0.0]/0.015, bins=30)
    # plt.hist(data_to_plot.to("cpu").numpy(), bins=100)
    plt.title("Histogram of Flattened Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"{PARENT}/test1.png")
        

if __name__=="__main__":
    main()