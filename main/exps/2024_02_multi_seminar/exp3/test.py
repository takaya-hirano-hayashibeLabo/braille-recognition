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

class DataTransformNrm():
    """
    ２次元データ（画像と同じ次元）をリサイズ＆正規化するクラス
    """
        
    def __call__(self,data,size=(28,28),max=None,min=None):
        """
        :param data: [N x C x H x W]
        :param size: 変換後のサイズ
        :return data_nrm, max, min
        """
        
        if not torch.is_tensor(data):
            data=torch.Tensor(data)
        
        if max is None and min is None:
            max=torch.max(data)
            min=torch.min(data)
            
        data_nrm=F.interpolate(
            torch.Tensor((data-min)/(1e-20+max)),
            size,mode='area'
        )
        
        return data_nrm,max,min


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--net_type",required=True,type=str)
    parser.add_argument("--model_name",type=str,required=True)
    parser.add_argument("--data_dir",required=True,help="3dデータが入っているディレクトリ")
    args=parser.parse_args()


    if args.net_type=="snn".casefold():
        MODEL_DIR="/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/train3d/snn_20240209_17.48.54"
    # MODEL_DIR=f"{str(PARENT.parent)}/models/simple_conv2d_"+args.net_type
    else:
        MODEL_DIR="/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/train3d/nn_20240209_19.33.35"


    var_sensornum_data=[]
    # data_dirs=os.listdir(args.data_dir)
    # print(data_dirs)
    # data_dirs.sort()
    # print(data_dirs)
    # exit(1)
    for dir_name in os.listdir(args.data_dir):

        #dir_nameからセンサーのサイズを取ってくる. dir_nameがdata_8x8_~の形式になっている前提
        sensor_size=int(dir_name.split("_")[1].split("x")[0])

        input_data:np.ndarray=np.load(f"{args.data_dir}/{dir_name}/input_3d.npy")
        input_data=input_data[:,:,np.newaxis,:,:] #channel方向に次元を伸ばす
        label_data:np.ndarray=np.load(f"{args.data_dir}/{dir_name}/label.npy").astype(int)

        # >> データのリサイズと標準化 >>
        mean,std=pd.read_csv(f"{MODEL_DIR}/std_param.csv").values[0]
        data_size=(64,64) #28, 64
        # transform=DataTransformStd()
        transform=DataTransformNrm()

        n,t,c,h,w=input_data.shape
        input_data_nrm,mean,std=transform(input_data.reshape(n*t,c,h,w),data_size,mean,std)
        input_data_nrm=input_data_nrm.view(n,t,c,data_size[0],data_size[1])

        var_sensornum_data.append(
            {"sensor_size":sensor_size,"input_data_nrm":input_data_nrm,"label_data":label_data}
        )


    #>> 学習パラメータの読み込み >>
    train_param={}
    with open(f"{MODEL_DIR}/train_param.txt", "r") as f:
        for line in f.readlines():
            key,val=line.replace("\n","").split(":",1)
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
    net.load_state_dict(torch.load(f"{MODEL_DIR}/{str(args.model_name).replace('.pth','')}.pth"))
    #>> ネットワークの読み込み >>


    save_dir=f"{PARENT}/result/{args.net_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f"{save_dir}/arg_params.json","w",encoding="utf-8_sig") as f:
        import json
        json.dump(args.__dict__,f,indent=4)


    result_list=[]
    net.eval()
    with torch.no_grad():
        for data in tqdm(var_sensornum_data):

            if args.net_type=="nn".casefold():
                out=net.forward(data["input_data_nrm"])

            elif args.net_type=="snn".casefold():
                #>> SNNは一気にやるとGPUのメモリが足りないので分割してやる >>
                split_num=5
                split_batch_size=int(data["input_data_nrm"].shape[0]/split_num)
                out=[]
                for i in range(split_num):
                    if i<split_num-1:
                        out_i=net.forward(data["input_data_nrm"][i*split_batch_size:i*split_batch_size+split_batch_size])
                    else:
                        out_i=net.forward(data["input_data_nrm"][i*split_batch_size:])
                    
                    if len(out)==0:
                        out=out_i
                    else:
                        out=torch.cat((out,out_i),dim=1)
                out=torch.sum(out,dim=0)
                #>> 一気にやるとメモリが足りないので分割してやる >>

            est=torch.argmax(out,dim=1).to("cpu").numpy()
            acc_vector=np.where(data["label_data"]==est,1,0) #正解ならTrue, ハズレならFalse

            result_j=[]
            for i in range(10):
                acc_vector_i=acc_vector[np.argwhere(data["label_data"]==i)]
                mean_i=np.mean(acc_vector_i) #accuracy計算
                result_j.append(mean_i)

            result_list.append([data["sensor_size"]]+result_j)

            result_table=pd.DataFrame(
                result_list,
                columns=["sensor_size"]+list(range(10)),
            )
            print(result_table)
            result_table.sort_values(by=["sensor_size"],inplace=True)
            result_table.to_csv(f"{save_dir}/result.csv",encoding="utf-8_sig",index=False)


            # アニメーション用の図と軸を準備
            fig, ax = plt.subplots()
            # アニメーションに追加するイメージのリストを作成
            images = []
            for i in range(t):  # tはタイムステップの数
                img = ax.imshow(np.fliplr(data['input_data_nrm'][0, i, 0].to("cpu").numpy()), animated=True,norm=Normalize(vmin=0, vmax=1))
                images.append([img])
            # ArtistAnimationのインスタンスを作成
            ani = arani(fig, images, interval=50, blit=True, repeat_delay=1000)
            # アニメーションを.mp4ファイルとして保存
            ani.save(f"{PARENT}/result/{args.net_type}/inputdata_sensorsize{data['sensor_size']}.mp4", dpi=200)
            plt.close()


    mu=np.mean(np.array(result_table.values)[:,1:],axis=1)
    sigma=np.std(np.array(result_table.values)[:,1:],axis=1)
    print(mu)
    plt.plot(
        result_table["sensor_size"].values.flatten(),
        mu,
        color="blue",marker="o",
        label=args.net_type  # legendにargs.net_typeを追加
    )
    plt.fill_between(
        x=result_table["sensor_size"].values.flatten(),
        y1=mu-sigma,y2=mu+sigma,
        color="blue",alpha=0.5
    )
    plt.xlabel("sensor num")  # xラベルをsensor numに設定
    plt.ylabel("accuracy")  # yラベルをaccuracyに設定
    plt.legend()  # legendを表示
    plt.ylim([0,1])
    plt.savefig(f"{save_dir}/fig.png")
        

if __name__=="__main__":
    main()