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

    model=mujoco.MjModel.from_xml_path(f'{PARENT}/simple_finger.xml') #xmlの読み込み
    data=mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model,data) as viewer:
        
        viewer.cam.type=0
        
        # >> handを動かすためのパラメータ >>
        # 点字が読める人は90 文字/分らしい (http://vips.eng.niigata-u.ac.jp/Tactile/TacPressWorkshop/ReviewInstruction.pdf)
        # １文字あたり5.40 mm(日本語)の幅がある
        # つまり, 点字習得者は1.5 mm/s の速度で指を動かしている.
        timescale=1
        v=timescale*1.5*1e-3 #[m/s] 指を動かす速度
        x=5.4*2* 1e-3 #[m] ハンドの初期位置
        idx_xpos=0 #qposにおけるハンドx座標のindex
        data.qpos[idx_xpos]=x
        # data.qpos[idx_zpos]=init_z
        row,col=128, 64
        # row,col=10, 20
        sensor_num=row*col
        # >> handを動かすためのパラメータ >>
        
        pressure_data=[]
        
        start_time=time.time()
        while data.time<15/timescale:
            
            step_start = time.time()
            
            new_qpos=deepcopy(data.qpos) #位置制御用のパラメータ

            #>> ハンドを動かす >>
            x-=v*model.opt.timestep
            new_qpos[idx_xpos]=x            
            #>> ハンドを動かす >>
            
            data.qpos=new_qpos #位置を更新
            mujoco.mj_step(model,data)
            
            print(time.time()-start_time)
            print(data.sensordata)
            print('-----')
            raw_data=deepcopy(data.sensordata[:sensor_num]) #1次元の生データ
            pressure_map=np.array(raw_data).reshape(row,col)
            
            pressure_data.append(pressure_map) #deepcopyしないと参照になる
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    pressure_data_std=(np.array(pressure_data)-np.min(pressure_data))/(np.max(pressure_data)-np.min(pressure_data))
    fig,ax=plt.subplots(1,1)
    ax.set_aspect("equal")
    frames=[]
    fps=6
    for t in range(pressure_data_std.shape[0]):
        if not (t%24)==0:
            continue
        pressure_digit=np.where(
            pressure_data_std[t]>np.mean(pressure_data_std),
            1,
            0
        )
        frame_i=[[ax.imshow(pressure_data_std[t], norm= Normalize(0,1))]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        # frame_i=[[ax.imshow(pressure_digit,cmap='gray',norm=Normalize(0,1))]+[ax.text(pressure_data_std.shape[1]*0.5,-1,s=f'elapesd_time:{model.opt.timestep*t}')]]
        # for x,y in zip(xx.flatten(),yy.flatten()):
        #     frame_i+=
        frames+=frame_i
        # frames+=[ax.plot(pressure_data_std[t],color='blue')+[ax.text(pressure_data_std.shape[1]*0.8,0.9,s=f'{model.opt.timestep*t}')]]
    ani=ArtistAnimation(fig,frames,interval=round(1000/fps))
    # ani.save(f'{PARENT}/pressure_map.gif')
    plt.show()
        
if __name__=="__main__":
    main()