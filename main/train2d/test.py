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

from train import Datasets,DataTransformStd,SNN,CNN

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--net_type",default='nn',type=str)
    parser.add_argument("--dot_shape",default="default",
                        type=str,help="dot_shape:{'default', 'sloped'}"
                        )
    args=parser.parse_args()
    
    
    if args.dot_shape=="default":
        data_dir=f"{ROOT}/data_collection/data2d"
    elif args.dot_shape=="sloped":
        data_dir=f"{ROOT}/data_collection/data2d-sloped"
        
    input_data:np.ndarray=np.load(f"{data_dir}/input_2d.npy")
    input_data=input_data[:,np.newaxis,:,:] #channel方向に次元を伸ばす
    label_data:np.ndarray=np.load(f"{data_dir}/label.npy").astype(int)

    # >> データのリサイズと標準化 >>
    data_shape=(64,64)
    transform=DataTransformStd()
    input_data_nrm,_,_=transform(input_data,data_shape)
    
    # plt.imshow(input_data_nrm.to("cpu")[300][0])
    # plt.show()
    # exit(1)
    # >> データのリサイズと標準化 >>
    
    #>> データのシャッフルと分割 >>
    train_size_rate=0.1
    train_size=round(train_size_rate*input_data.shape[0]) #学習データのサイズ
    print(f"train_size:{train_size}, test_size:{input_data.shape[0]-train_size}")
    shuffle_idx=torch.randperm(input_data.shape[0])
    
    train_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[:train_size]],
        y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)
        )
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]]*1,
        y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)
        )
    
    batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>

    
    #>> ネットワークの設定 >>
    if args.net_type=="nn".casefold():
        net=CNN()
    elif args.net_type=="snn".casefold():
        net=SNN(num_steps=24)
    else:
        print("net_type error")
        exit(1)
    net.load_state_dict(torch.load(f"{PARENT}/{args.net_type.upper()}/model.pth"))
    net.eval()
    #>> ネットワークの設定 >>
    
    
    test_acc=net.batch_accuracy(test_loader)
    print(f"Test Acc. : {test_acc.item()*100:.2f} %")
    

if __name__=="__main__":
    main()