from pathlib import Path
PARENT=str(Path(__file__).parent)

import mujoco
import mujoco.viewer
import time
import numpy as np
from copy import deepcopy
import os

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize

import argparse
import re

def main():
    
    """
    点字をコマンドラインから配置するプログラム
    点字の形, 点字の位置を設定できる
    """
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--braille_name",type=str,default='a')
    parser.add_argument("--braille_pos",type=str,default='0.033 -0.426 -0.136') #点字の初期位置
    # parser.add_argument("--braille_pos",type=str,default='0.033 -0.426 -0.1371785') #点字の初期位置
    parser.add_argument("--asset_path",type=str,default=f"{PARENT}/xml-resource/assets")
    args=parser.parse_args()
    
    
    # >> 点字オブジェクトの読み込み >>
    braille_xml_name=re.sub('.xml','',args.braille_name)+'.xml'
    with open(f'{PARENT}/xml-resource/{braille_xml_name}',"r") as f:
        braille_xml=f.readlines()
    braille_xml[0]=f"<body pos='{args.braille_pos}'>\n" #点字位置を上書き
    braille_xml="".join(braille_xml)
    braille_xml=re.sub('geom',"geom contype='2' conaffinity='2'",braille_xml) #接触タイプを追加
    # print(braille_xml)
    # >> 点字オブジェクトの読み込み >>
    
    # >>ハンドオブジェクトを読み込み >>
    with open(f"{PARENT}/xml-resource/finger_template.xml","r") as f:
        hand_xml="".join(f.readlines())
    # print(hand_xml)
    # >>ハンドオブジェクトを読み込み >>
    
    #>> 点字の配置 >>
    obj_xml=re.sub('{BRAILLE_BODY}',braille_xml,hand_xml)
    print(obj_xml)
    #>> 点字の配置 >>
    
    #>> assetの読み込み >>
    assets={}
    asset_files=os.listdir(f"{args.asset_path}")
    for asset_file in asset_files:
        with open(f"{args.asset_path}/{asset_file}", "rb") as f:
            assets[asset_file]=f.read()
    # print(assets)
    #>> assetの読み込み >>
    
    
    model=mujoco.MjModel.from_xml_string(obj_xml,assets) #xmlの読み込み
    data=mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model,data) as viewer:
        
        viewer.cam.type=0
        
        # >> handを動かすためのパラメータ >>
        # 点字が読める人は90 文字/分らしい (http://vips.eng.niigata-u.ac.jp/Tactile/TacPressWorkshop/ReviewInstruction.pdf)
        # １文字あたり5.40 mm(日本語)の幅がある
        # つまり, 点字習得者は1.5 mm/s の速度で指を動かしている.
        timescale=1.0
        v=timescale * 1.5*1e-3 #[m/s] 指を動かす速度
        init_x=5.4*2* 1e-3 #[m] ハンドの初期位置
        idx_xpos=1 #qposにおけるハンドx座標のindex
        data.qpos[idx_xpos]=init_x
        row,col=64,64
        sensor_num=row*col
        # >> handを動かすためのパラメータ >>
        
        pressure_data=[]
        
        while data.time<12/timescale:
            
            step_start = time.time()
            
            new_qpos=deepcopy(data.qpos) #位置制御用のパラメータ

            #>> 人差指の先を平行にする >>
            index_finger_params=[
                {'idx':3,'qpos':0},
                {'idx':4,'qpos':-1.22e-2},
                {'idx':5,'qpos':-6.89e-4},
                {'idx':6,'qpos':0.131},
            ]
            for param in index_finger_params:
                new_qpos[param['idx']]=param['qpos']
            #>> 人差指の先を平行にする >>

            #>> ハンドを動かす >>
            new_qpos[idx_xpos]-=v*model.opt.timestep            
            #>> ハンドを動かす >>
            
            data.qpos=new_qpos #位置を更新
            mujoco.mj_step(model,data)
            
            print(data.sensordata)
            raw_data=deepcopy(data.sensordata[:sensor_num]) #1次元の生データ
            pressure_map=np.array(raw_data).reshape(row,col)
            
            pressure_data.append(pressure_map) #deepcopyしないと参照になる
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # exit(1)
    xx,yy=np.meshgrid(range(row),range(col))
    pressure_data_std=(np.array(pressure_data)-np.min(pressure_data))/(np.max(pressure_data)-np.min(pressure_data))
    fig,ax=plt.subplots(1,1)
    frames=[]
    for t in range(pressure_data_std.shape[0]):
        pressure_digit=np.where(
            pressure_data_std[t]>np.mean(pressure_data_std),
            1,
            0
        )
        frame_i=[[ax.imshow(pressure_data_std[t],cmap='gray',norm=Normalize(0,1))]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        # frame_i=[[ax.imshow(pressure_digit,cmap='gray',norm=Normalize(0,1))]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        frames+=frame_i
        # frames+=[ax.plot(pressure_data_std[t],color='blue')+[ax.text(pressure_data_std.shape[1]*0.8,0.9,s=f'{model.opt.timestep*t}')]]
    ani=ArtistAnimation(fig,frames,interval=(model.opt.timestep*1000))
    # ani.save(f'{PARENT}/pressure_map.gif')
    plt.show()
        
if __name__=="__main__":
    main()