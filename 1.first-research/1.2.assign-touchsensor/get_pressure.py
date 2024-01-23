from pathlib import Path
PARENT=str(Path(__file__).parent)

import mujoco
import mujoco.viewer
import time
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import Normalize

def main():

    model=mujoco.MjModel.from_xml_path(f'{PARENT}/xml-resource/finger_with_touchsensor.xml') #xmlの読み込み
    data=mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model,data) as viewer:
        
        viewer.cam.type=0
        
        # >> handを動かすためのパラメータ >>
        # 点字が読める人は90 文字/分らしい (http://vips.eng.niigata-u.ac.jp/Tactile/TacPressWorkshop/ReviewInstruction.pdf)
        # １文字あたり5.40 mm(日本語)の幅がある
        # つまり, 点字習得者は1.5 mm/s の速度で指を動かしている.
        v=1.5*1e-3 #[m/s] 指を動かす速度
        init_x=5.4*2* 1e-3 #[m] ハンドの初期位置
        idx_xpos=1 #qposにおけるハンドx座標のindex
        data.qpos[idx_xpos]=init_x
        row,col=20,12
        sensor_num=row*col
        # >> handを動かすためのパラメータ >>
        
        pressure_data=[]
        
        while data.time<15:
            
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
            
            print(np.sum(data.sensordata))
            raw_data=deepcopy(data.sensordata[:sensor_num]) #1次元の生データ
            pressure_map=np.array(raw_data).reshape(row,col)
            
            pressure_data.append(pressure_map) #deepcopyしないと参照になる
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    xx,yy=np.meshgrid(range(row),range(col))
    pressure_data_std=(np.array(pressure_data)-np.min(pressure_data))/(np.max(pressure_data)-np.min(pressure_data))
    print(pressure_data_std)
    fig,ax=plt.subplots(1,1)
    frames=[]
    for t in range(pressure_data_std.shape[0]):
        pressure_digit=np.where(
            pressure_data_std[t]>np.mean(pressure_data_std),
            1,
            0
        )
        frame_i=[[ax.imshow(pressure_data_std[t],cmap='gray')]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        # frame_i=[[ax.imshow(pressure_digit,cmap='gray',norm=Normalize(0,1))]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        # for x,y in zip(xx.flatten(),yy.flatten()):
        #     frame_i+=
        frames+=frame_i
        # frames+=[ax.plot(pressure_data_std[t],color='blue')+[ax.text(pressure_data_std.shape[1]*0.8,0.9,s=f'{model.opt.timestep*t}')]]
    ani=ArtistAnimation(fig,frames,interval=(model.opt.timestep*1000))
    # ani.save(f'{PARENT}/pressure_map.gif')
    plt.show()
        
if __name__=="__main__":
    main()