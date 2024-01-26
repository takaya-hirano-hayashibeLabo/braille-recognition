"""
時系列学習させるときのデータ整形
SKIP_FRAMEごとのフレームを, TIME_SEQUENCEコ時間方向に並べたものを1つの入力とする.
"""

from pathlib import Path
PARENT=str(Path(__file__).parent)
import os

import numpy as np
import re
import argparse
from tqdm import tqdm


TIME_SEQUENCE=16 #16frame分の時系列データを畳み込む
SKIP_FRAME=20


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dot_shape",default="default",
                        type=str,help="dot_shape:{'default', 'sloped'}"
                        )
    args=parser.parse_args()
    
    dot_shape=args.dot_shape
    
    #>> simulationでとったデータのうち, この範囲内のデータを２次元データ（画像like）として保存>>
    time_start=8.4
    time_end=10.1
    #>> simulationでとったデータのうち, この範囲内のデータを２次元データ（画像like）として保存>>
    
    input_3d=[]
    label=[]
    
    if dot_shape=="default":
        save_dir=f"{PARENT}/data3d"
        org_data_dir=f"{PARENT}/data"
    elif dot_shape=="sloped":
        save_dir=f"{PARENT}/data3d-sloped"
        org_data_dir=f"{PARENT}/data-sloped"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for file in os.listdir(org_data_dir):
        if not "npy" in file:
            continue
        pressure_data=np.load(f"{org_data_dir}/{file}")
        
        file_name_tmp=file.split("_")
        braille_name=file_name_tmp[0]
        timestep=re.sub("timestep","",file_name_tmp[1])
        timestep=float(re.sub(".npy","",timestep))
        
        print(f"[braille :  {braille_name}]"+"="*50)

        # print(f"time_sequence [s] : {timestep*TIME_SEQUENCE*SKIP_FRAME} s")
        # print(pressure_data.shape)
        # exit(1)
        
        for t,data_t in tqdm(enumerate(pressure_data)):
            
            #>> ここの処理を変える >>
            if time_start <= t*timestep <=time_end:
                input_3d.append(pressure_data[t-(TIME_SEQUENCE*SKIP_FRAME)+1:t+1:SKIP_FRAME])
                label+=[braille_name]
            #>> ここの処理を変える >>
    
    input_3d=np.array(input_3d,dtype=np.float16)
    label=np.array(label)
    print(input_3d.shape)
    print(label.shape)
    
    np.save(f"{save_dir}/input_3d",input_3d)
    np.save(f"{save_dir}/label",label)
        
if __name__=="__main__":
    main()