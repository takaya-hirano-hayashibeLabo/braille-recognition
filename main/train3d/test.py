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
from matplotlib.animation import ArtistAnimation as arani
from matplotlib import gridspec
from matplotlib.colors import Normalize # Normalizeをimport
from tqdm import tqdm
import pandas as pd

from train import Datasets,DataTransformStd
from src import ECO,ECOSNN


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_dir",type=str,required=True)
    parser.add_argument("--model_name",type=str,required=True)
    parser.add_argument("--data_dir",default=f"{ROOT}/data_collection/data3d")
    args=parser.parse_args()

    input_data:np.ndarray=np.load(f"{args.data_dir}/input_3d.npy")
    input_data=input_data[:,np.newaxis,:,:] #channel方向に次元を伸ばす
    label_data:np.ndarray=np.load(f"{args.data_dir}/label.npy").astype(int)

    # >> データのリサイズと標準化 >>
    mean,std=pd.read_csv(f"{args.model_dir}/std_param.csv").values[0]
    data_size=(64,64) #28, 64
    transform=DataTransformStd()
    n,t,c,h,w=input_data.shape
    input_data_nrm,mean,std=transform(input_data.reshape(n*t,c,h,w),data_size,mean,std)
    input_data_nrm=input_data_nrm.view(n,t,c,data_size[0],data_size[1])

    #>> テストデータの選択 >>
    braille_char=8 #テストする点字
    input_data_char=input_data_nrm[label_data==braille_char]

    #>> 学習パラメータの読み込み >>
    train_param={}
    with open(f"{args.model_dir}/train_param.txt", "r") as f:
        for line in f.readlines():
            key,val=line.replace("\n","").split(":")
            train_param[key]=val
    # print(train_param)

    #>> ネットワークの読み込み >>
    if train_param["net_type"]=="nn".casefold():
        net=ECO()
    elif train_param["net_type"]=="snn".casefold():
        net=ECOSNN(
            snn_time_step=int(train_param["snn_steps"]),
            snn_threshold=float(train_param["snn_threshold"])
            )
    else:
        print("net_type error")
        exit(1)
    net.load_state_dict(torch.load(f"{args.model_dir}/{str(args.model_name).replace('.pth','')}.pth"))
    #>> ネットワークの読み込み >>


    fig=plt.figure(constrained_layout=True)
    gs=gridspec.GridSpec(3,4,figure=fig)
    ax=[
        fig.add_subplot(gs[:,0:3]),
        fig.add_subplot(gs[:,3])
    ]
    ax[1].set_yticks([i for i in range(10)])
    frames=[]
    net.eval()
    with torch.no_grad():

        input_data_tr=torch.transpose(torch.Tensor(input_data_char),1,2)
        for x,label in tqdm(zip(input_data_tr,label_data[label_data==braille_char])):
            
            # x+=torch.rand_like(x)
            out=net(x.unsqueeze(0))

            if train_param["net_type"]=="nn".casefold():
                class_prob=nn.functional.softmax(out)[0].to("cpu").numpy()
            elif train_param["net_type"]=="snn".casefold():
                class_prob=nn.functional.softmax(torch.sum(out,dim=0))[0].to("cpu").numpy()
            y=np.argmax(class_prob)
            # print(class_prob)
            # print(f"acutual : {label}, estimate : {y.item()}")

            x_sequence_map=np.sum([x[t].to("cpu").numpy()*(0.1+0.1*t) for t in range(x.shape[0])],axis=0)
            frames+=[[
                ax[0].imshow(
                    np.fliplr(x[-1][0].to("cpu").numpy()),aspect="auto",cmap="viridis",
                    norm=Normalize(vmin=torch.min(input_data_tr).item(),vmax=torch.max(input_data_tr).item()))
                ]+list(ax[1].barh(range(10),class_prob,align="center",color="blue"))]
            # print(frames)
            # exit(1)

    ani=arani(fig=fig,artists=frames,interval=100)
    ani.save(f"{args.model_dir}/test_anim.mp4")

if __name__=="__main__":
    main()