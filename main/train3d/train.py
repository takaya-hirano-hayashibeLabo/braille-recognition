from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent
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

import gc

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


def batch_accuracy(net:nn.Module,net_type:str,data_loader):
    with torch.no_grad():
        total=0
        acc=0
        net.eval()
        
        for data, targets in iter(data_loader):
            
            out:torch.Tensor= net(data)

            if net_type.casefold()=="nn":
                est_class=torch.argmax(out,dim=1) #選択されたクラス
                acc+=torch.sum(est_class==targets) #同じならTrueで１が合算される
                total += out.shape[0]

            elif net_type.casefold()=="snn":
                acc += SF.accuracy_rate(out, targets) * out.shape[1]
                total += out.shape[1]
                

            else:
                print("net_type error @fn:batch_accuracy")
                exit(1)

        if net_type.casefold()=="snn":
            print("firing rate"+"="*50)
            print(torch.sum(out,dim=0))
            print("label"+"="*50)
            print(targets)
            print("")
                
        return acc/total


def main():
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--net_type",default='nn',type=str)
    parser.add_argument("--save_dir",default=f"{PARENT}",type=str)
    args=parser.parse_args()
    
    data_dir=f"{ROOT}/main/data_collection/data3d"
    input_data:np.ndarray=np.load(f"{data_dir}/input_3d.npy")
    input_data=input_data[:,:,np.newaxis,:,:] #channel方向に次元を伸ばす
    label_data:np.ndarray=np.load(f"{data_dir}/label.npy").astype(int)

    # >> データのリサイズと標準化 >>
    data_size=(64,64) #28, 64
    transform=DataTransformStd()
    n,t,c,h,w=input_data.shape
    input_data_nrm,mean,std=transform(input_data.reshape(n*t,c,h,w),data_size)
    input_data_nrm=input_data_nrm.view(n,t,c,data_size[0],data_size[1])
    # >> データのリサイズと標準化 >>
    
    #>> データのシャッフルと分割 >>
    train_size_rate=0.7
    train_size=round(train_size_rate*input_data.shape[0]) #学習データのサイズ
    print(f"train_size:{train_size}, test_size:{input_data.shape[0]-train_size}")
    shuffle_idx=torch.randperm(input_data.shape[0])
    
    train_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[:train_size]],
        y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)
        )
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]],
        y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)
        )
    
    batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>
    
    #>> ネットワークの設定 >>
    if args.net_type=="nn".casefold():
        net=ECO()
        criterion=torch.nn.CrossEntropyLoss()
    elif args.net_type=="snn".casefold():
        net=ECOSNN(snn_time_step=24)
        criterion=SF.ce_rate_loss()
    else:
        print("net_type error")
        exit(1)
        
        
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    optimizer.param_groups[0]["capturable"]=True
    num_epochs = 30
    loss_hist = []
    test_acc_hist = []
    counter = 0
        
    # Outer training loop
    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(train_loader):
            
            # forward pass
            net.train()
            out = net.forward(data)

            # initialize the loss & sum over time
            loss_val:torch.Tensor = criterion(out, targets.type(torch.long))

            # Gradient calculation + weight update
            # print(f"{torch.cuda.memory_allocated()/(1024**3)} G") 
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 100 == 0:
                with torch.no_grad():
                    net.eval()
                    
                    # Train acc
                    train_acc=batch_accuracy(
                        net=net,net_type=args.net_type,
                        data_loader=train_loader
                    )

                    # Test set forward pass
                    test_acc=torch.randn(size=(1,))
                    # test_acc = batch_accuracy(
                    #     net=net,net_type=args.net_type,
                    #     data_loader=test_loader
                    # )
                    print(f"Iteration {counter}, Train Acc: {train_acc*100:.2f}%")# ,Test Acc: {test_acc * 100:.2f}%")
                    # print(f"Iteration {counter}, Train Acc: {train_acc*100:.2f}% ,Test Acc: {test_acc * 100:.2f}%")
                    test_acc_hist.append(test_acc.item())

            counter += 1
            
    #Save model
    save_dir=f"{PARENT}/{args.save_dir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(net.state_dict(),f"{save_dir}/model.pth")
    with open(f"{save_dir}/std_param.csv","w") as f:
        lines=f"mean,std\n{mean.item()},{std.item()}"
        f.write(lines)
            
    # Plot Loss
    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Test Set Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{save_dir}/test_accuracy")
    plt.show()
    
    if args.net_type=="snn".casefold():
        idx = 0

        fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
        labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']

        spikes=net(input_data_nrm)

        #  Plot spike count histogram
        anim = splt.spike_count(spikes[:, idx].detach().cpu(), fig, ax, labels=labels,
                                animate=True, interpolate=4)
        plt.show()
    
if __name__=="__main__":
    main()