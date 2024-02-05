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
import sys
import json


TIME_SEQUENCE=16 #16frame分の時系列データを畳み込む
SKIP_FRAME=20


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dot_shape",default="default",
                        type=str,help="dot_shape:{'default', 'sloped'}"
                        )
    parser.add_argument("--start_time",default=8.4,type=float)
    parser.add_argument("--end_time",default=10.1,type=float)
    parser.add_argument("--save_dir",required=True,help="RELATIVE path from this file")
    parser.add_argument("--data_dir",required=True,help="original data directry RELATIVE path from thie file")
    args=parser.parse_args()
    
    dot_shape=args.dot_shape
    
    #>> simulationでとったデータのうち, この範囲内のデータを２次元データ（画像like）として保存>>
    time_start=float(args.start_time)
    time_end=  float(args.end_time)
    #>> simulationでとったデータのうち, この範囲内のデータを２次元データ（画像like）として保存>>
    
    input_3d=[]
    label=[]
    
    org_data_dir=f"{PARENT}/{args.data_dir}"
    save_dir=f"{PARENT}"+f"/{args.save_dir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #>> コマンドライン引数の保存 >>
    with open(f"{save_dir}/params.json","w",encoding="utf-8") as f:
        json.dump(args.__dict__,f,indent=4)
    #>> コマンドライン引数の保存 >>
        
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
        
        prev_t=0
        for t,data_t in (enumerate(pressure_data)):
            
            #>> ここの処理を変えた >>
            if time_start <= t*timestep <=time_end:
                # print(prev_t,t)

                if t-prev_t>=(TIME_SEQUENCE*1) and (t-(TIME_SEQUENCE*SKIP_FRAME)+1)>=0: #ResNetは3フレーム分畳み込むので,3つは飛ばしていい
                    input_3d.append(pressure_data[t-(TIME_SEQUENCE*SKIP_FRAME)+1:t+1:SKIP_FRAME])
                    label+=[braille_name]
                    prev_t=t
                    
            elif t*timestep>time_end:
                break
            #>> ここの処理を変えた >>
        print(f"{sys.getsizeof(input_3d)/1e9} G")

    input_3d=np.array(input_3d,dtype=np.float16)
    label=np.array(label)
    print(input_3d.shape)
    print(label.shape)
    
    np.save(f"{save_dir}/input_3d",input_3d)
    np.save(f"{save_dir}/label",label)

        
if __name__=="__main__":
    main()