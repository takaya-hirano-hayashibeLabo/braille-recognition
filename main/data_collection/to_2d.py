from pathlib import Path
PARENT=str(Path(__file__).parent)
import os

import numpy as np
import re
import argparse

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
    
    input_2d=[]
    label=[]
    
    if dot_shape=="default":
        save_dir=f"{PARENT}/data2d"
        org_data_dir=f"{PARENT}/data"
    elif dot_shape=="sloped":
        save_dir=f"{PARENT}/data2d-sloped"
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
        
        # print(f"{braille_name}, {timestep}")
        # print(pressure_data.shape)
        
        for t,data_t in enumerate(pressure_data):
            
            if time_start <= t*timestep <=time_end:
                input_2d.append(data_t)
                label.append(braille_name)
                
    input_2d=np.array(input_2d)
    label=np.array(label)
    # print(input_2d.shape)
    # print(label.shape)
    
    np.save(f"{save_dir}/input_2d",input_2d)
    np.save(f"{save_dir}/label",label)
        
if __name__=="__main__":
    main()