from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent
import sys
sys.path.append(str(ROOT))
import os

import argparse

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import matplotlib.pyplot as plt

from train2d.train import Datasets,DataTransformStd,SNN,CNN

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
    with open(f"{ROOT}/train2d/{args.net_type.upper()}/std_param.csv","r") as f:
        lines=f.readlines()
        mean,std=[float(param) for param in lines[1].split(",")]
    print(f"mean:{mean}, std:{std}")
    data_shape=(64,64)
    transform=DataTransformStd()
    input_data_nrm,_,_=transform(input_data,data_shape,mean=mean,std=std)
    
    # plt.imshow(input_data_nrm.to("cpu")[300][0])
    # plt.show()
    # exit(1)
    # >> データのリサイズと標準化 >>
        
    #>> ネットワークの設定 >>
    if args.net_type=="nn".casefold():
        net=CNN()
    elif args.net_type=="snn".casefold():
        net=SNN(num_steps=24)
    else:
        print("net_type error")
        exit(1)
    net.load_state_dict(torch.load(f"{str(ROOT)}/train2d/{args.net_type.upper()}/model.pth"))
    net.eval()
    #>> ネットワークの設定 >>
    
    
    # frac_list=[0.1]
    frac_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    result_table=[]
    
    with torch.no_grad():
        for frac in tqdm(frac_list):
            if args.net_type=="nn".casefold():
                out=net.forward(input_data_nrm*frac)
            elif args.net_type=="snn".casefold():
                out=torch.sum(net.forward(input_data_nrm*frac),dim=0)
            est=torch.argmax(out,dim=1)
            
            fracs=np.zeros_like(label_data)+frac
            table=np.concatenate(
                [fracs.reshape(-1,1),
                label_data.reshape(-1,1),
                est.to("cpu").detach().numpy().reshape(-1,1)],
                axis=1
            )
            
            if len(result_table)==0:
                result_table=table
            else:
                result_table=np.concatenate([result_table,table],axis=0)
        
    result_table=pd.DataFrame(
        result_table,columns=["frac","label","estimate"]
    )
    result_table.astype({"label":int})
    result_table.astype({"estimate":int})
    result_table.to_csv(f"{PARENT}/result_{args.net_type}.csv",encoding="utf-8",index=False)

if __name__=="__main__":
    main()