from pathlib import Path
PARENT=str(Path(__file__).parent)
import os

import mujoco
import mujoco.viewer
import time
import numpy as np
from copy import deepcopy
import re
import xmltodict
from tqdm import tqdm
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize


class Simulator():
        
    def __init__(self,sim_env_xml=None):
        if sim_env_xml is None:
            with open(f"{PARENT}/assets/sim_env_init.xml", "r") as f:
                self.sim_env_xml_init="".join(f.readlines())
            self.braille_name=""
                
    
    def laod_braille_body(self,braille_name='a'):
        """
        点字bodyをシミュレーション環境xmlに読み込む
        :param braille_name:読み込む点字のローマ字名
        """
        
        self.braille_name=braille_name
        
        #>> 点字bodyを読み込んで, 位置と回転を合わせる >>
        with open(f"{PARENT}/assets/braille-xml/{braille_name}.xml", "r") as f:
            braille_body=f.readlines()
        for i,line in enumerate(braille_body):
            if "!" in line:
                continue
            top_line=line
            idx_top=i
            break
        braille_body[idx_top]=re.sub("pos='0 0 0'", "pos='0.033 -0.425 -0.144'  axisangle='0 0 1 3.14'", top_line)
        braille_body="".join(braille_body)
        braille_body=re.sub(
            "geom ",
            "geom contype='2' conaffinity='2' ",
            braille_body
        )
        #>> 点字bodyを読み込んで, 位置と回転を合わせる >>
        
        
        #>> シミュレーションxmlと点字bodyを合体 >>
        self.sim_env_xml=re.sub(
            "{BRAILLE_BODY}",
            braille_body,
            self.sim_env_xml_init
        )
        with open(f"{PARENT}/assets/sim_env_tmp.xml", "w") as f:
            f.write(self.sim_env_xml)
        #>> シミュレーションxmlと点字bodyを合体 >>
        
        
        
    def simulate(self,is_view=False,hand_v=1.5*1e-3,hand_x=5.4*2*1e-3,time_th=15,save_dir=f"{PARENT}"):
        """
        シミュレーションしてタッチマップを取得する関数
        """
        
        model=mujoco.MjModel.from_xml_path(f'{PARENT}/assets/sim_env_tmp.xml') #xmlの読み込み
        data=mujoco.MjData(model)
        
        pressure_data=[]
        
        #>> touch sensorの縦横のサイズをXMLから抽出 >>
        xml_root=xmltodict.parse(self.sim_env_xml)
        for attr in xml_root['mujoco']['sensor']['plugin']['config']:
            if attr['@key']=='size':
                sensor_col,sensor_row=[int(val) for val in attr['@value'].split()]
        # print(sensor_row,sensor_col)
        #>> touch sensorの縦横のサイズをXMLから抽出 >>
        
        print(f"===SIMULATE BRAILLE {self.braille_name}===")
        
        # >> 描画するとき >>
        if is_view:
            with mujoco.viewer.launch_passive(model,data) as viewer:
                while data.time<time_th:
                    
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
                    hand_x-=hand_v*model.opt.timestep
                    new_qpos[1]=hand_x
                    #>> ハンドを動かす >>
                    
                    data.qpos=new_qpos #位置を更新
                    mujoco.mj_step(model,data)
                    
                    raw_data=deepcopy(data.sensordata) #1次元の生データ
                    pressure_map=np.array(raw_data).reshape(sensor_row,sensor_col)
                    
                    pressure_data.append(pressure_map) #deepcopyしないと参照になる
                    
                    viewer.sync()
                    
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
            
            
        # >> 描画しないとき >>
        elif not is_view:
            tqdm_total=ceil(time_th/model.opt.timestep)+1
            with tqdm(total=tqdm_total) as pbar:
                while data.time<time_th:
                        
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
                    hand_x-=hand_v*model.opt.timestep
                    new_qpos[1]=hand_x
                    #>> ハンドを動かす >>
                    
                    data.qpos=new_qpos #位置を更新
                    mujoco.mj_step(model,data)
                    
                    raw_data=deepcopy(data.sensordata) #1次元の生データ
                    pressure_map=np.array(raw_data).reshape(sensor_row,sensor_col)
                    
                    pressure_data.append(pressure_map) #deepcopyしないと参照になる
                    
                    pbar.update(1)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        #>> マップに描画 >>
        fig,ax=plt.subplots(1,1)
        frames=[]
        fps=24
        for t in range(np.array(pressure_data).shape[0]):
            if not (t%fps)==0:
                continue
            frame_i=[[ax.imshow(np.fliplr(pressure_data[t]))]+[ax.text(np.array(pressure_data).shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
            frames+=frame_i
        ani=ArtistAnimation(fig,frames,interval=round(1000/fps))
        ani.save(f'{save_dir}/{self.braille_name}_timestep{model.opt.timestep}.mp4')
        plt.close()
        # plt.show()
        #>> マップに描画 >>
            
        np.save(f"{save_dir}/{self.braille_name}_timestep{model.opt.timestep}",
                np.array(pressure_data))
    
    
        
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',type=str)
    args=parser.parse_args()
    
    simulator=Simulator()
    simulator.laod_braille_body(braille_name=args.name)
    simulator.simulate(is_view=True)